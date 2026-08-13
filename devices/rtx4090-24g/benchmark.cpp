#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#error "RTX4090 benchmark harness currently requires Linux."
#endif

namespace fs=std::filesystem;

namespace qingming::benchmark {

constexpr int kRuns=10;

static void print_progress(
    const char* mode,
    int completed,
    const char* phase="") {

    constexpr int width=20;

    const int bounded=
        std::max(
            0,
            std::min(
                completed,
                kRuns));

    const int filled=
        bounded*width/kRuns;

    std::cout
        <<"\rMode: "
        <<std::left
        <<std::setw(8)
        <<mode
        <<std::right
        <<" [";

    for(int index=0;index<width;++index){
        std::cout
            <<(index<filled?'=':'-');
    }

    std::cout
        <<"] "
        <<bounded
        <<"/"
        <<kRuns;

    if(
        phase!=nullptr
        &&phase[0]!='\0'
    ){
        std::cout
            <<"  "
            <<phase;
    }

    // Padding overwrites remnants from a longer previous status line.
    std::cout
        <<"                                                                "
        <<std::flush;

    if(bounded==kRuns){
        std::cout
            <<"\n"
            <<std::flush;
    }
}

struct Options {
    std::string family;
    std::string task;
    std::string text_mode;
    std::string speaker;
    std::string instruct;
    fs::path binary;
    fs::path model_dir;
    fs::path ref_audio;
    fs::path output_dir;
    fs::path report_json;
    fs::path report_txt;
    std::string language="English";
    std::string text;
    std::uint64_t seed=1234;
    std::size_t max_new_tokens=0;
    bool max_new_tokens_explicit=false;
};

static std::size_t generation_capacity_for(
    std::size_t requested) {

    if(requested==0||requested>8192){
        throw std::runtime_error(
            "max_new_tokens must be in [1,8192]");
    }

    std::size_t capacity=256;
    while(capacity<requested){
        capacity<<=1;
    }
    return capacity;
}

struct ProcessOutput {
    int exit_code=-1;
    std::string output;
    double wall_ms=0.0;
};

struct Result {
    bool ok=false;
    bool eos=false;
    int frames=-1;
    double duration_s=-1.0;
    double predictor_p50_ms=-1.0;
    double predictor_p95_ms=-1.0;
    double ttft_ms=-1.0;
    double ttfa_ms=-1.0;
    double generation_ms=-1.0;
    double decoder_ms=-1.0;
    double e2e_ms=-1.0;
    double realtime_x=-1.0;
    double process_wall_ms=-1.0;
    std::uint64_t wav_fnv1a64=0;
    std::string codec_fnv1a64;
    std::uintmax_t wav_bytes=0;
    fs::path output_wav;
    std::string raw;
};

struct Stats {
    double min=0.0;
    double mean=0.0;
    double p50=0.0;
    double p95=0.0;
    double max=0.0;
};

static std::string json_escape(
    const std::string& input) {

    std::ostringstream output;

    for(const unsigned char c:input){
        switch(c){
            case '"': output<<"\\\""; break;
            case '\\': output<<"\\\\"; break;
            case '\b': output<<"\\b"; break;
            case '\f': output<<"\\f"; break;
            case '\n': output<<"\\n"; break;
            case '\r': output<<"\\r"; break;
            case '\t': output<<"\\t"; break;
            default:
                if(c<0x20){
                    output
                        <<"\\u"
                        <<std::hex
                        <<std::setw(4)
                        <<std::setfill('0')
                        <<static_cast<unsigned>(c)
                        <<std::dec
                        <<std::setfill(' ');
                }else{
                    output<<static_cast<char>(c);
                }
                break;
        }
    }

    return output.str();
}

static std::optional<std::string>
json_string(
    const std::string& line,
    const std::string& key) {

    const std::string needle=
        std::string("\"")+key+"\":\"";

    const auto begin=line.find(needle);

    if(begin==std::string::npos){
        return std::nullopt;
    }

    std::size_t position=
        begin+needle.size();

    std::string value;

    while(position<line.size()){
        const char c=line[position++];

        if(c=='"'){
            return value;
        }

        if(c!='\\'){
            value.push_back(c);
            continue;
        }

        if(position>=line.size()){
            break;
        }

        const char e=line[position++];

        switch(e){
            case '"': value.push_back('"'); break;
            case '\\': value.push_back('\\'); break;
            case '/': value.push_back('/'); break;
            case 'b': value.push_back('\b'); break;
            case 'f': value.push_back('\f'); break;
            case 'n': value.push_back('\n'); break;
            case 'r': value.push_back('\r'); break;
            case 't': value.push_back('\t'); break;
            default:
                value.push_back(e);
                break;
        }
    }

    return std::nullopt;
}

static std::optional<double>
json_number(
    const std::string& line,
    const std::string& key) {

    const std::string needle=
        std::string("\"")+key+"\":";    

    const auto begin=line.find(needle);

    if(begin==std::string::npos){
        return std::nullopt;
    }

    const char* first=
        line.c_str()
        +begin
        +needle.size();

    char* end=nullptr;

    const double value=
        std::strtod(
            first,
            &end);

    if(end==first){
        return std::nullopt;
    }

    return value;
}

static bool json_bool(
    const std::string& line,
    const std::string& key,
    bool fallback=false) {

    const std::string needle=
        std::string("\"")+key+"\":";    

    const auto begin=line.find(needle);

    if(begin==std::string::npos){
        return fallback;
    }

    const auto position=
        begin+needle.size();

    if(
        line.compare(
            position,
            4,
            "true")==0
    ){
        return true;
    }

    if(
        line.compare(
            position,
            5,
            "false")==0
    ){
        return false;
    }

    return fallback;
}

static std::string find_json_result(
    const std::string& output) {

    std::istringstream stream(output);
    std::string line;
    std::string found;

    while(std::getline(stream,line)){
        if(
            line.find("\"status\":")
                !=std::string::npos
            &&line.find("\"device\":\"rtx4090-24g\"")
                !=std::string::npos
        ){
            found=line;
        }
    }

    if(found.empty()){
        // Resident output may be prefixed by "> ".
        const auto object=
            output.rfind("{\"status\":");

        if(object!=std::string::npos){
            const auto end=
                output.find(
                    '\n',
                    object);

            found=
                output.substr(
                    object,
                    end==std::string::npos
                    ?std::string::npos
                    :end-object);
        }
    }

    return found;
}

static std::uint64_t fnv1a64_file(
    const fs::path& path) {

    std::ifstream stream(
        path,
        std::ios::binary);

    if(!stream){
        return 0;
    }

    std::uint64_t hash=
        1469598103934665603ULL;

    std::array<char,1<<16> buffer{};

    while(stream){
        stream.read(
            buffer.data(),
            static_cast<std::streamsize>(
                buffer.size()));

        const auto got=
            stream.gcount();

        for(std::streamsize i=0;i<got;++i){
            hash^=
                static_cast<unsigned char>(
                    buffer[
                        static_cast<std::size_t>(i)]);

            hash*=
                1099511628211ULL;
        }
    }

    return hash;
}

static Result parse_result(
    const std::string& raw,
    double process_wall_ms) {

    Result result;
    result.raw=raw;
    result.process_wall_ms=process_wall_ms;

    const std::string line=
        find_json_result(raw);

    if(line.empty()){
        return result;
    }

    const auto status=
        json_string(
            line,
            "status");

    result.ok=
        status
        &&*status=="ok";

    result.eos=
        json_bool(
            line,
            "eos",
            false);

    if(
        const auto value=
            json_number(
                line,
                "frames")
    ){
        result.frames=
            static_cast<int>(*value);
    }

    auto set_double=
        [&](const char* key,double& target){
            if(
                const auto value=
                    json_number(
                        line,
                        key)
            ){
                target=*value;
            }
        };

    set_double(
        "audio_duration_s",
        result.duration_s);

    set_double(
        "predictor_p50_ms",
        result.predictor_p50_ms);

    set_double(
        "predictor_p95_ms",
        result.predictor_p95_ms);

    set_double(
        "ttft_ms",
        result.ttft_ms);

    set_double(
        "ttfa_ms",
        result.ttfa_ms);

    set_double(
        "generation_ms",
        result.generation_ms);

    set_double(
        "decoder_ms",
        result.decoder_ms);

    set_double(
        "e2e_ms",
        result.e2e_ms);

    set_double(
        "realtime_x",
        result.realtime_x);

    if(
        const auto codec_hash=
            json_string(
                line,
                "codec_fnv1a64")
    ){
        result.codec_fnv1a64=*codec_hash;
    }

    if(
        const auto output=
            json_string(
                line,
                "output")
    ){
        result.output_wav=
            fs::path(*output);

        std::error_code error;

        if(
            fs::is_regular_file(
                result.output_wav,
                error)
        ){
            result.wav_bytes=
                fs::file_size(
                    result.output_wav,
                    error);

            if(!error){
                result.wav_fnv1a64=
                    fnv1a64_file(
                        result.output_wav);
            }
        }
    }

    return result;
}

static ProcessOutput run_process(
    const std::vector<std::string>& args) {

    if(args.empty()){
        throw std::runtime_error(
            "empty process command");
    }

    int pipe_fd[2];

    if(pipe(pipe_fd)!=0){
        throw std::runtime_error(
            "pipe failed");
    }

    const auto begin=
        std::chrono::steady_clock::now();

    const pid_t pid=fork();

    if(pid<0){
        close(pipe_fd[0]);
        close(pipe_fd[1]);

        throw std::runtime_error(
            "fork failed");
    }

    if(pid==0){
        close(pipe_fd[0]);

        dup2(
            pipe_fd[1],
            STDOUT_FILENO);

        dup2(
            pipe_fd[1],
            STDERR_FILENO);

        close(pipe_fd[1]);

        std::vector<char*> argv;
        argv.reserve(args.size()+1);

        for(const auto& arg:args){
            argv.push_back(
                const_cast<char*>(
                    arg.c_str()));
        }

        argv.push_back(nullptr);

        execv(
            argv[0],
            argv.data());

        _exit(127);
    }

    close(pipe_fd[1]);

    std::string output;
    std::array<char,4096> buffer{};

    for(;;){
        const ssize_t got=
            read(
                pipe_fd[0],
                buffer.data(),
                buffer.size());

        if(got>0){
            output.append(
                buffer.data(),
                static_cast<std::size_t>(got));
            continue;
        }

        break;
    }

    close(pipe_fd[0]);

    int status=0;
    waitpid(pid,&status,0);

    const auto end=
        std::chrono::steady_clock::now();

    ProcessOutput result;
    result.output=std::move(output);
    result.wall_ms=
        std::chrono::duration<double,std::milli>(
            end-begin)
            .count();

    if(WIFEXITED(status)){
        result.exit_code=
            WEXITSTATUS(status);
    }

    return result;
}

class ResidentProcess {
public:
    explicit ResidentProcess(
        const std::vector<std::string>& args) {

        if(args.empty()){
            throw std::runtime_error(
                "empty resident command");
        }

        int input_pipe[2];
        int output_pipe[2];

        if(
            pipe(input_pipe)!=0
            ||pipe(output_pipe)!=0
        ){
            throw std::runtime_error(
                "resident pipe failed");
        }

        const auto begin=
            std::chrono::steady_clock::now();

        pid_=fork();

        if(pid_<0){
            throw std::runtime_error(
                "resident fork failed");
        }

        if(pid_==0){
            close(input_pipe[1]);
            close(output_pipe[0]);

            (void)setenv(
                "QINGMING_BENCHMARK_TRACE",
                "1",
                1);

            dup2(
                input_pipe[0],
                STDIN_FILENO);

            dup2(
                output_pipe[1],
                STDOUT_FILENO);

            dup2(
                output_pipe[1],
                STDERR_FILENO);

            close(input_pipe[0]);
            close(output_pipe[1]);

            std::vector<char*> argv;
            argv.reserve(args.size()+1);

            for(const auto& arg:args){
                argv.push_back(
                    const_cast<char*>(
                        arg.c_str()));
            }

            argv.push_back(nullptr);

            execv(
                argv[0],
                argv.data());

            _exit(127);
        }

        close(input_pipe[0]);
        close(output_pipe[1]);

        input_fd_=input_pipe[1];
        output_fd_=output_pipe[0];

        const std::string startup=
            read_until_prompt(
                0,
                60000,
                false);

        (void)startup;

        const auto ready=
            std::chrono::steady_clock::now();

        startup_ms_=
            std::chrono::duration<double,std::milli>(
                ready-begin)
                .count();
    }

    ~ResidentProcess() {
        if(input_fd_>=0){
            close(input_fd_);
        }

        if(output_fd_>=0){
            close(output_fd_);
        }

        if(pid_>0){
            int status=0;
            waitpid(
                pid_,
                &status,
                WNOHANG);
        }
    }

    double startup_ms()const{
        return startup_ms_;
    }

    ProcessOutput request(
        const std::string& json,
        int completed_before) {

        const auto begin=
            std::chrono::steady_clock::now();

        write_all(
            json+"\n");

        const std::string output=
            read_until_prompt(
                completed_before,
                30000,
                true);

        const auto end=
            std::chrono::steady_clock::now();

        ProcessOutput result;
        result.exit_code=0;
        result.output=output;
        result.wall_ms=
            std::chrono::duration<double,std::milli>(
                end-begin)
                .count();

        return result;
    }

    void shutdown() {
        if(input_fd_<0){
            return;
        }

        write_all(
            "{\"command\":\"shutdown\"}\n");

        close(input_fd_);
        input_fd_=-1;

        if(output_fd_>=0){
            std::array<char,1024> buffer{};

            while(
                read(
                    output_fd_,
                    buffer.data(),
                    buffer.size())>0
            ){}

            close(output_fd_);
            output_fd_=-1;
        }

        if(pid_>0){
            int status=0;
            waitpid(pid_,&status,0);
            pid_=-1;
        }
    }

private:
    pid_t pid_=-1;
    int input_fd_=-1;
    int output_fd_=-1;
    double startup_ms_=0.0;

    void write_all(
        const std::string& text) {

        const char* data=text.data();
        std::size_t remaining=text.size();

        while(remaining>0){
            const ssize_t written=
                write(
                    input_fd_,
                    data,
                    remaining);

            if(written<=0){
                throw std::runtime_error(
                    "resident stdin write failed");
            }

            data+=written;
            remaining-=
                static_cast<std::size_t>(
                    written);
        }
    }

    static std::string latest_stage(
        const std::string& output) {

        const std::string marker=
            "@resident_stage request=";

        const auto begin=
            output.rfind(marker);

        if(begin==std::string::npos){
            return {};
        }

        const auto stage_pos=
            output.find(
                " stage=",
                begin);

        if(stage_pos==std::string::npos){
            return {};
        }

        const auto value_begin=
            stage_pos
            +std::strlen(
                " stage=");

        const auto value_end=
            output.find(
                '\n',
                value_begin);

        return output.substr(
            value_begin,
            value_end==std::string::npos
                ?std::string::npos
                :value_end-value_begin);
    }

    [[noreturn]]
    void timeout_fail(
        int completed_before,
        const std::string& last_stage) {

        if(pid_>0){
            (void)kill(
                pid_,
                SIGKILL);

            int status=0;
            (void)waitpid(
                pid_,
                &status,
                0);

            pid_=-1;
        }

        if(input_fd_>=0){
            close(input_fd_);
            input_fd_=-1;
        }

        if(output_fd_>=0){
            close(output_fd_);
            output_fd_=-1;
        }

        throw std::runtime_error(
            "Resident request "
            +std::to_string(
                completed_before+1)
            +" timeout; last stage="
            +(
                last_stage.empty()
                ?std::string("unknown")
                :last_stage
            ));
    }

    std::string read_until_prompt(
        int completed_before,
        int timeout_ms,
        bool show_stage) {

        std::string output;
        std::array<char,4096> buffer{};
        std::string last_stage;

        const auto deadline=
            std::chrono::steady_clock::now()
            +std::chrono::milliseconds(
                timeout_ms);

        for(;;){
            const auto now=
                std::chrono::steady_clock::now();

            if(now>=deadline){
                timeout_fail(
                    completed_before,
                    last_stage);
            }

            const auto remaining=
                std::chrono::duration_cast<
                    std::chrono::milliseconds>(
                        deadline-now)
                    .count();

            pollfd descriptor{};
            descriptor.fd=output_fd_;
            descriptor.events=
                POLLIN|POLLHUP|POLLERR;

            const int ready=
                ::poll(
                    &descriptor,
                    1,
                    static_cast<int>(
                        std::min<long long>(
                            remaining,
                            1000)));

            if(ready<0){
                if(errno==EINTR){
                    continue;
                }

                throw std::runtime_error(
                    "resident poll failed");
            }

            if(ready==0){
                continue;
            }

            const ssize_t got=
                read(
                    output_fd_,
                    buffer.data(),
                    buffer.size());

            if(got<=0){
                throw std::runtime_error(
                    "resident process ended before prompt; last stage="
                    +(
                        last_stage.empty()
                        ?std::string("unknown")
                        :last_stage
                    ));
            }

            output.append(
                buffer.data(),
                static_cast<std::size_t>(
                    got));

            const std::string stage=
                latest_stage(output);

            if(
                !stage.empty()
                &&stage!=last_stage
            ){
                last_stage=stage;

                if(show_stage){
                    const std::string phase=
                        "request "
                        +std::to_string(
                            completed_before+1)
                        +": "
                        +last_stage;

                    print_progress(
                        "Resident",
                        completed_before,
                        phase.c_str());
                }
            }

            if(
                output.size()>=2
                &&output.compare(
                    output.size()-2,
                    2,
                    "> ")==0
            ){
                return output;
            }
        }
    }
};

static Stats stats(
    std::vector<double> values) {

    values.erase(
        std::remove_if(
            values.begin(),
            values.end(),
            [](double value){
                return
                    !std::isfinite(value)
                    ||value<0.0;
            }),
        values.end());

    if(values.empty()){
        return {};
    }

    std::sort(
        values.begin(),
        values.end());

    auto percentile=
        [&](double p){
            if(values.size()==1){
                return values.front();
            }

            const double position=
                p*(
                    static_cast<double>(
                        values.size()-1));

            const std::size_t lower=
                static_cast<std::size_t>(
                    std::floor(position));

            const std::size_t upper=
                static_cast<std::size_t>(
                    std::ceil(position));

            const double fraction=
                position-lower;

            return
                values[lower]
                +(
                    values[upper]
                    -values[lower]
                )*fraction;
        };

    Stats result;
    result.min=values.front();
    result.max=values.back();
    result.mean=
        std::accumulate(
            values.begin(),
            values.end(),
            0.0)
        /static_cast<double>(
            values.size());
    result.p50=percentile(0.50);
    result.p95=percentile(0.95);

    return result;
}

static std::vector<double>
metric(
    const std::vector<Result>& results,
    double Result::*member) {

    std::vector<double> values;
    values.reserve(results.size());

    for(const auto& result:results){
        values.push_back(
            result.*member);
    }

    return values;
}

static bool all_success(
    const std::vector<Result>& results) {

    return
        results.size()==kRuns
        &&std::all_of(
            results.begin(),
            results.end(),
            [](const Result& result){
                return result.ok;
            });
}

static bool all_eos(
    const std::vector<Result>& results) {

    return
        results.size()==kRuns
        &&std::all_of(
            results.begin(),
            results.end(),
            [](const Result& result){
                return result.eos;
            });
}

static bool frames_consistent(
    const std::vector<Result>& results) {

    if(results.empty()){
        return false;
    }

    const int expected=
        results.front().frames;

    return
        expected>0
        &&std::all_of(
            results.begin(),
            results.end(),
            [&](const Result& result){
                return
                    result.frames==expected;
            });
}

static bool hashes_consistent(
    const std::vector<Result>& results) {

    if(results.empty()){
        return false;
    }

    const auto expected=
        results.front().wav_fnv1a64;

    if(expected==0){
        return false;
    }

    return
        std::all_of(
            results.begin(),
            results.end(),
            [&](const Result& result){
                return
                    result.wav_fnv1a64
                    ==expected;
            });
}

static bool codec_hashes_consistent(
    const std::vector<Result>& results) {

    if(results.empty()){
        return false;
    }

    const auto& expected=
        results.front().codec_fnv1a64;

    if(expected.empty()){
        return false;
    }

    return
        std::all_of(
            results.begin(),
            results.end(),
            [&](const Result& result){
                return result.codec_fnv1a64==expected;
            });
}

static void write_stats_json(
    std::ostream& stream,
    const char* name,
    const Stats& value,
    bool comma=true) {

    stream
        <<"    \""<<name<<"\": {"
        <<"\"min\":"<<value.min<<","
        <<"\"mean\":"<<value.mean<<","
        <<"\"p50\":"<<value.p50<<","
        <<"\"p95\":"<<value.p95<<","
        <<"\"max\":"<<value.max
        <<"}"
        <<(comma?",":"")
        <<"\n";
}

static void print_stats_text(
    std::ostream& stream,
    const char* name,
    const Stats& value) {

    stream
        <<std::left
        <<std::setw(18)
        <<name
        <<std::right
        <<" min "
        <<std::setw(9)
        <<value.min
        <<"  mean "
        <<std::setw(9)
        <<value.mean
        <<"  p50 "
        <<std::setw(9)
        <<value.p50
        <<"  p95 "
        <<std::setw(9)
        <<value.p95
        <<"  max "
        <<std::setw(9)
        <<value.max
        <<"\n";
}

static Options parse_cli(
    int argc,
    char** argv) {

    Options options;

    for(int i=1;i<argc;++i){
        const std::string arg=argv[i];

        auto require=
            [&](const char* name){
                if(i+1>=argc){
                    throw std::runtime_error(
                        std::string(name)
                        +" requires a value");
                }

                return std::string(argv[++i]);
            };

        if(arg=="--family"){
            options.family=
                require("--family");
        }else if(arg=="--task"){
            options.task=
                require("--task");
        }else if(arg=="--text-mode"){
            options.text_mode=
                require("--text-mode");
        }else if(arg=="--speaker"){
            options.speaker=
                require("--speaker");
        }else if(arg=="--instruct"){
            options.instruct=
                require("--instruct");
        }else if(arg=="--binary"){
            options.binary=
                fs::path(
                    require("--binary"));
        }else if(arg=="--model-dir"){
            options.model_dir=
                fs::path(
                    require("--model-dir"));
        }else if(arg=="--ref-audio"){
            options.ref_audio=
                fs::path(
                    require("--ref-audio"));
        }else if(arg=="--language"){
            options.language=
                require("--language");
        }else if(arg=="--text"){
            options.text=
                require("--text");
        }else if(arg=="--seed"){
            options.seed=
                static_cast<std::uint64_t>(
                    std::stoull(
                        require("--seed")));
        }else if(arg=="--max-new-tokens"){
            options.max_new_tokens=
                static_cast<std::size_t>(
                    std::stoull(
                        require("--max-new-tokens")));
            options.max_new_tokens_explicit=true;
        }else if(arg=="--output-dir"){
            options.output_dir=
                fs::path(
                    require("--output-dir"));
        }else if(arg=="--report-json"){
            options.report_json=
                fs::path(
                    require("--report-json"));
        }else if(arg=="--report-txt"){
            options.report_txt=
                fs::path(
                    require("--report-txt"));
        }else if(
            arg=="--help"
            ||arg=="-h"
        ){
            std::cout
                <<"Usage:\n  "
                <<argv[0]
                <<" --family 1.7b|0.6b"
                <<" --task base-xvector|custom-voice|voice-design"
                <<" --text-mode streaming"
                <<" --model-dir DIR"
                <<" --language English"
                <<" --text TEXT"
                <<" [--ref-audio WAV]"
                <<" [--speaker Ryan]"
                <<" [--instruct TEXT]"
                <<" [--seed 1234]"
                <<" --max-new-tokens N"
                <<" [--output-dir DIR]"
                <<" [--report-json FILE]"
                <<" [--report-txt FILE]"
                <<"\n\n"
                <<"Runs exactly 10 Once requests and 10 Resident requests.\n";
            std::exit(0);
        }else{
            throw std::runtime_error(
                "unknown argument: "
                +arg);
        }
    }

    if(options.family.empty()){
        throw std::runtime_error(
            "--family is required");
    }

    if(
        options.family!="1.7b"
        &&options.family!="0.6b"
    ){
        throw std::runtime_error(
            "--family must be 1.7b or 0.6b");
    }

    if(
        options.task!="base-xvector"
        &&options.task!="custom-voice"
        &&options.task!="voice-design"
    ){
        throw std::runtime_error(
            "--task must be explicitly base-xvector, custom-voice, or voice-design");
    }

    if(options.text_mode!="streaming"){
        throw std::runtime_error(
            "--text-mode streaming is required");
    }

    if(
        options.family=="0.6b"
        &&options.task=="voice-design"
    ){
        throw std::runtime_error(
            "voice-design requires --family 1.7b");
    }

    if(!options.max_new_tokens_explicit){
        throw std::runtime_error(
            "--max-new-tokens is required");
    }
    if(options.max_new_tokens==0||options.max_new_tokens>8192){
        throw std::runtime_error(
            "--max-new-tokens must be in [1,8192]");
    }

    if(options.binary.empty()){
        const fs::path self=
            fs::absolute(
                argv[0]);

        options.binary=
            self.parent_path()
            /(
                "qingming-qwen3-tts_rtx4090-24g_"
                +options.family
            );
    }

    if(options.output_dir.empty()){
        options.output_dir=
            fs::path(
                "benchmark-rtx4090-24g-"
                +options.family
                +"-"
                +options.task);
    }

    if(options.report_json.empty()){
        options.report_json=
            fs::path(
                "benchmark-rtx4090-24g-"
                +options.family
                +"-"
                +options.task
                +".json");
    }

    if(options.report_txt.empty()){
        options.report_txt=
            fs::path(
                "benchmark-rtx4090-24g-"
                +options.family
                +"-"
                +options.task
                +".txt");
    }

    if(options.model_dir.empty()){
        throw std::runtime_error(
            "--model-dir is required");
    }

    if(
        options.task=="base-xvector"
        &&options.ref_audio.empty()
    ){
        throw std::runtime_error(
            "--ref-audio is required for base-xvector");
    }

    if(
        options.task=="custom-voice"
        &&options.speaker.empty()
    ){
        throw std::runtime_error(
            "--speaker is required for custom-voice");
    }

    if(
        options.task=="custom-voice"
        &&!options.ref_audio.empty()
    ){
        throw std::runtime_error(
            "--ref-audio is forbidden for custom-voice");
    }

    if(
        options.task=="base-xvector"
        &&(!options.speaker.empty()||!options.instruct.empty())
    ){
        throw std::runtime_error(
            "--speaker and --instruct are forbidden for base-xvector");
    }

    if(
        options.family=="0.6b"
        &&options.task=="custom-voice"
        &&!options.instruct.empty()
    ){
        throw std::runtime_error(
            "--instruct is not supported by 0.6b custom-voice");
    }

    if(options.task=="voice-design"){
        if(options.instruct.empty()){
            throw std::runtime_error(
                "--instruct is required for voice-design");
        }
        if(!options.ref_audio.empty()||!options.speaker.empty()){
            throw std::runtime_error(
                "--ref-audio and --speaker are forbidden for voice-design");
        }
    }

    if(options.text.empty()){
        throw std::runtime_error(
            "--text is required");
    }

    options.binary=
        fs::absolute(
            options.binary);

    options.model_dir=
        fs::absolute(
            options.model_dir);

    if(!options.ref_audio.empty()){
        options.ref_audio=
            fs::absolute(
                options.ref_audio);
    }

    options.output_dir=
        fs::absolute(
            options.output_dir);

    options.report_json=
        fs::absolute(
            options.report_json);

    options.report_txt=
        fs::absolute(
            options.report_txt);

    return options;
}

static std::string resident_request_json(
    const Options& options,
    const fs::path& output) {

    std::ostringstream request;

    request
        <<"{"
        <<"\"text\":\""
        <<json_escape(options.text)
        <<"\","
        <<"\"language\":\""
        <<json_escape(options.language)
        <<"\",";

    if(options.task=="base-xvector"){
        request
            <<"\"ref_audio\":\""
            <<json_escape(
                options.ref_audio.string())
            <<"\",";
    }else if(options.task=="custom-voice"){
        request
            <<"\"speaker\":\""
            <<json_escape(
                options.speaker)
            <<"\",";
    }

    if(!options.instruct.empty()){
        request
            <<"\"instruct\":\""
            <<json_escape(options.instruct)
            <<"\",";
    }

    request
        <<"\"output\":\""
        <<json_escape(
            output.string())
        <<"\","
        <<"\"seed\":"
        <<options.seed
        <<","
        <<"\"max_new_tokens\":"
        <<options.max_new_tokens
        <<"}";

    return request.str();
}

static int run(
    int argc,
    char** argv) {

    const Options options=
        parse_cli(
            argc,
            argv);

    if(!fs::is_regular_file(options.binary)){
        throw std::runtime_error(
            "production binary not found: "
            +options.binary.string());
    }

    fs::remove_all(
        options.output_dir);

    fs::create_directories(
        options.output_dir);

    std::vector<Result> once;
    std::vector<Result> resident;

    once.reserve(kRuns);
    resident.reserve(kRuns);

    std::cout
        <<"RTX4090-24G benchmark\n"
        <<"Family: "<<options.family<<"\n"
        <<"Task: "<<options.task<<"\n"
        <<"Text mode: "<<options.text_mode<<"\n"
        <<"Speaker: "<<options.speaker<<"\n"
        <<"Instruct: "<<(options.instruct.empty()?"n/a":options.instruct)<<"\n"
        <<"Runs: Once 10 + Resident 10\n"
        <<"Resident split: 80 SM generation / 48 SM decoder\n\n";

    // ---------------------------------------------------------------------
    // Once: launch the FORMAL binary 10 independent times.
    // ---------------------------------------------------------------------
    print_progress(
        "Once",
        0);
    for(int run_index=0;
        run_index<kRuns;
        ++run_index){

        const fs::path output=
            options.output_dir
            /(
                "once_"
                +std::to_string(
                    run_index+1)
                +".wav"
            );

        std::vector<std::string> args{
            options.binary.string(),
            "--lifecycle","once",
            "--model-dir",options.model_dir.string(),
            "--language",options.language,
            "--text",options.text,
            "--output",output.string(),
            "--max-new-tokens",
                std::to_string(
                    options.max_new_tokens),
            "--seed",
                std::to_string(
                    options.seed),
        };

        args.push_back("--task");
        args.push_back(options.task);
        args.push_back("--text-mode");
        args.push_back(options.text_mode);

        if(options.task=="base-xvector"){
            args.push_back("--ref-audio");
            args.push_back(
                options.ref_audio.string());
        }else if(options.task=="custom-voice"){
            args.push_back("--speaker");
            args.push_back(
                options.speaker);
        }

        if(!options.instruct.empty()){
            args.push_back("--instruct");
            args.push_back(options.instruct);
        }

        const auto process=
            run_process(args);

        Result result=
            parse_result(
                process.output,
                process.wall_ms);

        once.push_back(result);

        print_progress(
            "Once",
            run_index+1,
            result.ok
                ?""
                :"FAIL");
    }

    // ---------------------------------------------------------------------
    // Resident: load the FORMAL binary once, then issue 10 requests.
    // ---------------------------------------------------------------------
    print_progress(
        "Resident",
        0,
        "loading");
    std::vector<std::string> resident_args{
        options.binary.string(),
        "--lifecycle","resident",
        "--model-dir",options.model_dir.string(),
        "--max-new-tokens",
            std::to_string(
                options.max_new_tokens),
    };

    resident_args.push_back("--task");
    resident_args.push_back(options.task);
    resident_args.push_back("--text-mode");
    resident_args.push_back(options.text_mode);

    ResidentProcess resident_process(
        resident_args);

    const double resident_startup_ms=
        resident_process.startup_ms();

    print_progress(
        "Resident",
        0);

    for(int run_index=0;
        run_index<kRuns;
        ++run_index){

        const fs::path output=
            options.output_dir
            /(
                "resident_"
                +std::to_string(
                    run_index+1)
                +".wav"
            );

        const auto process=
            resident_process.request(
                resident_request_json(
                    options,
                    output),
                run_index);

        Result result=
            parse_result(
                process.output,
                process.wall_ms);

        resident.push_back(result);

        print_progress(
            "Resident",
            run_index+1,
            result.ok
                ?""
                :"FAIL");
    }

    resident_process.shutdown();

    std::cout
        <<"\n";

    const bool once_success=
        all_success(once);

    const bool resident_success=
        all_success(resident);

    const bool once_eos=
        all_eos(once);

    const bool resident_eos=
        all_eos(resident);

    const bool once_frames=
        frames_consistent(once);

    const bool resident_frames=
        frames_consistent(resident);

    const bool once_hash=
        hashes_consistent(once);

    const bool resident_hash=
        hashes_consistent(resident);

    const bool codec_hash_supported=true;

    const bool once_codec_hash=
        !codec_hash_supported
        ||codec_hashes_consistent(once);

    const bool resident_codec_hash=
        !codec_hash_supported
        ||codec_hashes_consistent(resident);

    const bool cross_codec_hash=
        !codec_hash_supported
        ||(
            !once.empty()
            &&!resident.empty()
            &&!once.front().codec_fnv1a64.empty()
            &&once.front().codec_fnv1a64
                ==resident.front().codec_fnv1a64
        );

    const bool cross_hash=
        !once.empty()
        &&!resident.empty()
        &&once.front().wav_fnv1a64!=0
        &&once.front().wav_fnv1a64
            ==resident.front().wav_fnv1a64;

    const bool trajectory_accuracy_pass=
        once_success
        &&resident_success
        &&once_eos
        &&resident_eos
        &&once_frames
        &&resident_frames
        &&once.front().frames
            ==resident.front().frames
        &&once_codec_hash
        &&resident_codec_hash
        &&cross_codec_hash;

    const auto once_ttft=
        stats(
            metric(
                once,
                &Result::ttft_ms));

    const auto once_ttfa=
        stats(
            metric(
                once,
                &Result::ttfa_ms));

    const auto once_generation=
        stats(
            metric(
                once,
                &Result::generation_ms));

    const auto once_decoder=
        stats(
            metric(
                once,
                &Result::decoder_ms));

    const auto once_e2e=
        stats(
            metric(
                once,
                &Result::e2e_ms));

    const auto once_wall=
        stats(
            metric(
                once,
                &Result::process_wall_ms));

    const auto resident_ttft=
        stats(
            metric(
                resident,
                &Result::ttft_ms));

    const auto resident_ttfa=
        stats(
            metric(
                resident,
                &Result::ttfa_ms));

    const auto resident_generation=
        stats(
            metric(
                resident,
                &Result::generation_ms));

    const auto resident_decoder=
        stats(
            metric(
                resident,
                &Result::decoder_ms));

    const auto resident_e2e=
        stats(
            metric(
                resident,
                &Result::e2e_ms));

    const auto resident_wall=
        stats(
            metric(
                resident,
                &Result::process_wall_ms));

    // Text report.
    std::ostringstream text_report;

    text_report
        <<std::fixed
        <<std::setprecision(3)
        <<"Qingming Qwen3-TTS RTX4090-24G Benchmark Report\n"
        <<"==================================================\n"
        <<"Family: "<<options.family<<"\n"
        <<"Task: "<<options.task<<"\n"
        <<"Text mode: "<<options.text_mode<<"\n"
        <<"Speaker: "<<options.speaker<<"\n"
        <<"Instruct: "<<(options.instruct.empty()?"n/a":options.instruct)<<"\n"
        <<"Runs per lifecycle: 10\n"
        <<"Resident SM split: generation 80 / decoder 48\n"
        <<"Seed: "<<options.seed<<"\n"
        <<"Max new tokens requested: "<<options.max_new_tokens<<"\n"
        <<"Max new tokens capacity: "
        <<generation_capacity_for(options.max_new_tokens)<<"\n"
        <<"First audio chunk frames: 8\n"
        <<"Steady streaming chunk frames: 16\n"
        <<"Model: "<<options.model_dir.string()<<"\n"
        <<"Reference WAV: "
        <<(
            options.ref_audio.empty()
            ?std::string{"n/a"}
            :options.ref_audio.string()
        )
        <<"\n\n"
        <<"ACCURACY\n"
        <<"--------\n"
        <<"Once success: "
        <<std::count_if(
            once.begin(),
            once.end(),
            [](const Result& r){return r.ok;})
        <<"/10\n"
        <<"Resident success: "
        <<std::count_if(
            resident.begin(),
            resident.end(),
            [](const Result& r){return r.ok;})
        <<"/10\n"
        <<"Once all EOS: "<<(once_eos?"PASS":"FAIL")<<"\n"
        <<"Resident all EOS: "<<(resident_eos?"PASS":"FAIL")<<"\n"
        <<"Once frame consistency: "<<(once_frames?"PASS":"FAIL")<<"\n"
        <<"Resident frame consistency: "<<(resident_frames?"PASS":"FAIL")<<"\n"
        <<"Cross lifecycle frame match: "
        <<(
            !once.empty()
            &&!resident.empty()
            &&once.front().frames
                ==resident.front().frames
            ?"PASS":"FAIL")
        <<"\n"
        <<"Once codec trajectory consistency: "
        <<(codec_hash_supported?(once_codec_hash?"PASS":"FAIL"):"N/A")<<"\n"
        <<"Resident codec trajectory consistency: "
        <<(codec_hash_supported?(resident_codec_hash?"PASS":"FAIL"):"N/A")<<"\n"
        <<"Once vs Resident codec trajectory exact: "
        <<(codec_hash_supported?(cross_codec_hash?"PASS":"FAIL"):"N/A")<<"\n"
        <<"Once WAV byte consistency: "<<(once_hash?"PASS":"FAIL")<<"\n"
        <<"Resident WAV byte consistency: "<<(resident_hash?"PASS":"FAIL")<<"\n"
        <<"Once vs Resident WAV byte exact: "<<(cross_hash?"PASS":"FAIL")<<"\n"
        <<"Trajectory accuracy: "<<(trajectory_accuracy_pass?"PASS":"FAIL")<<"\n";

    if(!once.empty()){
        text_report
            <<"Codec frames: "
            <<once.front().frames
            <<"\n";
    }

    text_report
        <<"\nPERFORMANCE - ONCE (ms)\n"
        <<"-----------------------\n";

    print_stats_text(
        text_report,
        "TTFT",
        once_ttft);

    print_stats_text(
        text_report,
        "TTFA",
        once_ttfa);

    print_stats_text(
        text_report,
        "Generation",
        once_generation);

    print_stats_text(
        text_report,
        "Decoder",
        once_decoder);

    print_stats_text(
        text_report,
        "E2E",
        once_e2e);

    print_stats_text(
        text_report,
        "Process wall",
        once_wall);

    text_report
        <<"\nPERFORMANCE - RESIDENT (ms)\n"
        <<"---------------------------\n"
        <<"Resident startup/load wall: "
        <<resident_startup_ms
        <<"\n";

    print_stats_text(
        text_report,
        "TTFT",
        resident_ttft);

    print_stats_text(
        text_report,
        "TTFA",
        resident_ttfa);

    print_stats_text(
        text_report,
        "Generation",
        resident_generation);

    print_stats_text(
        text_report,
        "Decoder",
        resident_decoder);

    print_stats_text(
        text_report,
        "E2E",
        resident_e2e);

    print_stats_text(
        text_report,
        "Request wall",
        resident_wall);

    text_report
        <<"\nOUTPUT\n"
        <<"------\n"
        <<"WAV directory: "
        <<options.output_dir.string()
        <<"\n"
        <<"JSON report: "
        <<options.report_json.string()
        <<"\n";

    // JSON report.
    std::ostringstream json_report;

    json_report
        <<std::fixed
        <<std::setprecision(6)
        <<"{\n"
        <<"  \"device\": \"rtx4090-24g\",\n"
        <<"  \"family\": \""<<json_escape(options.family)<<"\",\n"
        <<"  \"task\": \""<<json_escape(options.task)<<"\",\n"
        <<"  \"text_mode\": \""<<json_escape(options.text_mode)<<"\",\n"
        <<"  \"speaker\": \""<<json_escape(options.speaker)<<"\",\n"
        <<"  \"instruct\": \""<<json_escape(options.instruct)<<"\",\n"
        <<"  \"max_new_tokens\": "<<options.max_new_tokens<<",\n"
        <<"  \"max_new_tokens_capacity\": "
        <<generation_capacity_for(options.max_new_tokens)<<",\n"
        <<"  \"runs_per_lifecycle\": 10,\n"
        <<"  \"resident_generation_wgp\": 28,\n"
        <<"  \"resident_decoder_wgp\": 20,\n"
        <<"  \"seed\": "<<options.seed<<",\n"
        <<"  \"accuracy\": {\n"
        <<"    \"once_success\": "<<(once_success?"true":"false")<<",\n"
        <<"    \"resident_success\": "<<(resident_success?"true":"false")<<",\n"
        <<"    \"once_all_eos\": "<<(once_eos?"true":"false")<<",\n"
        <<"    \"resident_all_eos\": "<<(resident_eos?"true":"false")<<",\n"
        <<"    \"once_frames_consistent\": "<<(once_frames?"true":"false")<<",\n"
        <<"    \"resident_frames_consistent\": "<<(resident_frames?"true":"false")<<",\n"
        <<"    \"once_codec_trajectory_consistent\": "<<(once_codec_hash?"true":"false")<<",\n"
        <<"    \"resident_codec_trajectory_consistent\": "<<(resident_codec_hash?"true":"false")<<",\n"
        <<"    \"once_vs_resident_codec_trajectory_exact\": "<<(cross_codec_hash?"true":"false")<<",\n"
        <<"    \"once_wav_bytes_consistent\": "<<(once_hash?"true":"false")<<",\n"
        <<"    \"resident_wav_bytes_consistent\": "<<(resident_hash?"true":"false")<<",\n"
        <<"    \"once_vs_resident_wav_byte_exact\": "<<(cross_hash?"true":"false")<<",\n"
        <<"    \"trajectory_accuracy_pass\": "<<(trajectory_accuracy_pass?"true":"false")<<"\n"
        <<"  },\n"
        <<"  \"once_ms\": {\n";

    write_stats_json(
        json_report,
        "ttft",
        once_ttft);

    write_stats_json(
        json_report,
        "ttfa",
        once_ttfa);

    write_stats_json(
        json_report,
        "generation",
        once_generation);

    write_stats_json(
        json_report,
        "decoder",
        once_decoder);

    write_stats_json(
        json_report,
        "e2e",
        once_e2e);

    write_stats_json(
        json_report,
        "process_wall",
        once_wall,
        false);

    json_report
        <<"  },\n"
        <<"  \"resident_startup_ms\": "
        <<resident_startup_ms
        <<",\n"
        <<"  \"resident_ms\": {\n";

    write_stats_json(
        json_report,
        "ttft",
        resident_ttft);

    write_stats_json(
        json_report,
        "ttfa",
        resident_ttfa);

    write_stats_json(
        json_report,
        "generation",
        resident_generation);

    write_stats_json(
        json_report,
        "decoder",
        resident_decoder);

    write_stats_json(
        json_report,
        "e2e",
        resident_e2e);

    write_stats_json(
        json_report,
        "request_wall",
        resident_wall,
        false);

    json_report
        <<"  }\n"
        <<"}\n";

    fs::create_directories(
        options.report_txt.parent_path());

    fs::create_directories(
        options.report_json.parent_path());

    {
        std::ofstream stream(
            options.report_txt);

        stream<<text_report.str();
    }

    {
        std::ofstream stream(
            options.report_json);

        stream<<json_report.str();
    }

    std::cout
        <<"\n"
        <<text_report.str()
        <<"\n";

    return
        trajectory_accuracy_pass
        ?0
        :2;
}

} // namespace qingming::benchmark

int main(int argc,char** argv){
    try{
        return qingming::benchmark::
            run(
                argc,
                argv);
    }catch(const std::exception& error){
        std::cerr
            <<"FATAL: "
            <<error.what()
            <<"\n";
        return 1;
    }
}
