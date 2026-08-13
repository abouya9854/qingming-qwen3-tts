#if defined(QINGMING_MODEL_FAMILY_1_7B)

#define main qingming_qwen3_tts_rx7900xtx_24g_reference_main
#include "devices/rx7900xtx-24g/qwen3_tts_1_7b.cpp"
#undef main

namespace qingming::production {

struct CapturedIo {
    std::ostringstream out;
    std::ostringstream err;
    std::streambuf* old_out=nullptr;
    std::streambuf* old_err=nullptr;

    void begin() {
        old_out=std::cout.rdbuf(out.rdbuf());
        old_err=std::cerr.rdbuf(err.rdbuf());
    }

    void end() {
        if(old_out!=nullptr){
            std::cout.rdbuf(old_out);
            old_out=nullptr;
        }

        if(old_err!=nullptr){
            std::cerr.rdbuf(old_err);
            old_err=nullptr;
        }
    }

    ~CapturedIo() {
        end();
    }
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

static std::optional<double> find_number(
    const std::string& text,
    const std::string& key) {

    const std::string needle=key+":";
    const std::size_t begin=text.rfind(needle);

    if(begin==std::string::npos){
        return std::nullopt;
    }

    std::size_t position=begin+needle.size();

    while(
        position<text.size()
        &&std::isspace(
            static_cast<unsigned char>(
                text[position]))
    ){
        ++position;
    }

    const char* first=text.c_str()+position;
    char* end=nullptr;
    const double value=std::strtod(first,&end);

    if(end==first){
        return std::nullopt;
    }

    return value;
}

static std::optional<long long> find_integer(
    const std::string& text,
    const std::string& key) {

    const auto value=
        find_number(
            text,
            key);

    if(!value){
        return std::nullopt;
    }

    return static_cast<long long>(*value);
}

static std::optional<std::string> find_scalar_text(
    const std::string& text,
    const std::string& key) {

    const std::string needle=key+":";
    const std::size_t begin=text.rfind(needle);
    if(begin==std::string::npos){
        return std::nullopt;
    }

    std::size_t position=begin+needle.size();
    while(
        position<text.size()
        &&std::isspace(
            static_cast<unsigned char>(text[position]))
    ){
        ++position;
    }

    const std::size_t start=position;
    while(
        position<text.size()
        &&!std::isspace(
            static_cast<unsigned char>(text[position]))
    ){
        ++position;
    }

    if(position==start){
        return std::nullopt;
    }
    return text.substr(start,position-start);
}

static bool find_eos(
    const std::string& text) {

    return
        text.find(
            "legacy_native_recurrence_stop_c: reason=eos frame=")
        !=std::string::npos;
}

static std::optional<long long> find_eos_frame(
    const std::string& text) {

    const std::string needle=
        "legacy_native_recurrence_stop_c: reason=eos frame=";

    const std::size_t position=
        text.rfind(needle);

    if(position==std::string::npos){
        return std::nullopt;
    }

    const char* first=
        text.c_str()
        +position
        +needle.size();

    char* end=nullptr;

    const long long frame=
        std::strtoll(
            first,
            &end,
            10);

    if(end==first){
        return std::nullopt;
    }

    return frame;
}

static void print_error_result(
    const std::string& lifecycle,
    const std::string& message) {

    std::cout
        <<"{"
        <<"\"status\":\"error\","
        <<"\"device\":\"rx7900xtx-24g\","
        <<"\"family\":\"1.7b\","
        <<"\"execution_profile\":\"1.7b-rdna3-gfx1100\","
        <<"\"lifecycle\":\""
        <<json_escape(lifecycle)
        <<"\","
        <<"\"error\":\""
        <<json_escape(message)
        <<"\""
        <<"}\n"
        <<std::flush;
}

static void print_once_result(
    const std::string& captured,
    const std::string& captured_error,
    const std::string& output,
    const std::string& task,
    const std::string& text_mode,
    const std::string& speaker,
    std::uint64_t seed,
    std::size_t max_new_tokens,
    int rc) {

    const auto frames=
        find_integer(
            captured,
            "legacy_codec_frame_count_b");

    const auto codec_hash=
        find_scalar_text(
            captured,
            "legacy_codec_fnv1a64_b");

    const auto duration=
        find_number(
            captured,
            "legacy_audio_duration_s_b");

    const auto ttft=
        find_number(
            captured,
            "legacy_ttft_ms_b");

    const auto ttfa=
        find_number(
            captured,
            "legacy_ttfa_ms_b");

    const auto first_audio_chunk_frames=
        find_integer(
            captured,
            "streaming_decoder_first_chunk_frames");

    const auto streaming_chunk_frames=
        find_integer(
            captured,
            "streaming_decoder_chunk_frames");

    const auto generation=
        find_number(
            captured,
            "legacy_codec_generation_ms_b");

    const auto decoder=
        find_number(
            captured,
            "legacy_codec_decoder_graph_ms_b");

    const auto e2e=
        find_number(
            captured,
            "legacy_e2e_ms_b");

    const auto realtime=
        find_number(
            captured,
            "legacy_e2e_realtime_x_b");

    const auto p50=
        find_number(
            captured,
            "predictor_frame_p50_ms");

    const auto p95=
        find_number(
            captured,
            "predictor_frame_p95_ms");

    const bool eos=find_eos(captured);
    const auto eos_frame=find_eos_frame(captured);

    const bool ok=
        rc==0
        &&frames.has_value()
        &&e2e.has_value();

    std::cout
        <<std::fixed
        <<std::setprecision(3)
        <<"{"
        <<"\"status\":\""
        <<(ok?"ok":"error")
        <<"\","
        <<"\"device\":\"rx7900xtx-24g\","
        <<"\"family\":\"1.7b\","
        <<"\"execution_profile\":\"1.7b-rdna3-gfx1100\","
        <<"\"lifecycle\":\"once\","
        <<"\"task\":\""<<json_escape(task)<<"\","
        <<"\"text_mode\":\""<<json_escape(text_mode)<<"\","
        <<"\"speaker\":\""<<json_escape(speaker)<<"\","
        <<"\"error\":\""
        <<json_escape(
            ok
            ?std::string{}
            :captured_error)
        <<"\","
        <<"\"seed\":"<<seed<<","
        <<"\"max_new_tokens\":"<<max_new_tokens<<","
        <<"\"max_new_tokens_capacity\":"
        <<([&](){
            if(max_new_tokens==0||max_new_tokens>8192)return std::size_t{0};
            std::size_t capacity=256;
            while(capacity<max_new_tokens)capacity<<=1;
            return capacity;
        })()<<","
        <<"\"frames\":"<<(frames?*frames:-1)<<","
        <<"\"codec_fnv1a64\":\""
        <<json_escape(codec_hash?*codec_hash:std::string{})
        <<"\","
        <<"\"eos\":"<<(eos?"true":"false")<<","
        <<"\"eos_frame\":"<<(eos_frame?*eos_frame:-1)<<","
        <<"\"audio_duration_s\":"<<(duration?*duration:-1.0)<<","
        <<"\"predictor_p50_ms\":"<<(p50?*p50:-1.0)<<","
        <<"\"predictor_p95_ms\":"<<(p95?*p95:-1.0)<<","
        <<"\"ttft_ms\":"<<(ttft?*ttft:-1.0)<<","
        <<"\"ttfa_ms\":"<<(ttfa?*ttfa:-1.0)<<","
        <<"\"first_audio_chunk_frames\":"
        <<(first_audio_chunk_frames?*first_audio_chunk_frames:-1)<<","
        <<"\"first_audio_chunk_duration_ms\":"
        <<(first_audio_chunk_frames?*first_audio_chunk_frames*80.0:-1.0)<<","
        <<"\"streaming_chunk_frames\":"
        <<(streaming_chunk_frames?*streaming_chunk_frames:-1)<<","
        <<"\"generation_ms\":"<<(generation?*generation:-1.0)<<","
        <<"\"decoder_ms\":"<<(decoder?*decoder:-1.0)<<","
        <<"\"e2e_ms\":"<<(e2e?*e2e:-1.0)<<","
        <<"\"realtime_x\":"<<(realtime?*realtime:-1.0)<<","
        <<"\"output\":\""<<json_escape(output)<<"\""
        <<"}\n"
        <<std::flush;
}

static void print_resident_result(
    const std::string& captured,
    const qwen3_tts::baseline::ResidentRequest& request,
    const std::string& task,
    const std::string& text_mode) {

    const auto frames=
        find_integer(
            captured,
            "resident_codec_frame_count");

    const auto codec_hash=
        find_scalar_text(
            captured,
            "resident_codec_fnv1a64");

    const auto duration=
        find_number(
            captured,
            "resident_audio_duration_s");

    const auto ttft=
        find_number(
            captured,
            "resident_ttft_ms");

    const auto ttfa=
        find_number(
            captured,
            "resident_ttfa_ms");

    const auto first_audio_chunk_frames=
        find_integer(
            captured,
            "resident_streaming_decoder_first_chunk_frames");

    const auto streaming_chunk_frames=
        find_integer(
            captured,
            "resident_streaming_decoder_chunk_frames");

    const auto generation=
        find_number(
            captured,
            "resident_generation_ms");

    const auto decoder=
        find_number(
            captured,
            "resident_decoder_compute_ms");

    const auto e2e=
        find_number(
            captured,
            "resident_e2e_ms");

    const auto realtime=
        find_number(
            captured,
            "resident_e2e_realtime_x");

    const auto p50=
        find_number(
            captured,
            "resident_predictor_frame_p50_ms");

    const auto p95=
        find_number(
            captured,
            "resident_predictor_frame_p95_ms");

    const bool eos=find_eos(captured);
    const auto eos_frame=find_eos_frame(captured);

    const bool ok=
        frames.has_value()
        &&e2e.has_value()
        &&captured.find(
            "resident_request_success: True")
            !=std::string::npos;

    std::cout
        <<std::fixed
        <<std::setprecision(3)
        <<"{"
        <<"\"status\":\""
        <<(ok?"ok":"error")
        <<"\","
        <<"\"device\":\"rx7900xtx-24g\","
        <<"\"family\":\"1.7b\","
        <<"\"execution_profile\":\"1.7b-rdna3-gfx1100\","
        <<"\"lifecycle\":\"resident\","
        <<"\"task\":\""<<json_escape(task)<<"\","
        <<"\"text_mode\":\""<<json_escape(text_mode)<<"\","
        <<"\"speaker\":\""<<json_escape(request.speaker)<<"\","
        <<"\"seed\":"<<request.seed<<","
        <<"\"max_new_tokens\":"<<request.max_new_tokens<<","
        <<"\"max_new_tokens_capacity\":"
        <<([&](){
            if(request.max_new_tokens==0||request.max_new_tokens>8192)return std::size_t{0};
            std::size_t capacity=256;
            while(capacity<request.max_new_tokens)capacity<<=1;
            return capacity;
        })()<<","
        <<"\"frames\":"<<(frames?*frames:-1)<<","
        <<"\"codec_fnv1a64\":\""
        <<json_escape(codec_hash?*codec_hash:std::string{})
        <<"\","
        <<"\"eos\":"<<(eos?"true":"false")<<","
        <<"\"eos_frame\":"<<(eos_frame?*eos_frame:-1)<<","
        <<"\"audio_duration_s\":"<<(duration?*duration:-1.0)<<","
        <<"\"predictor_p50_ms\":"<<(p50?*p50:-1.0)<<","
        <<"\"predictor_p95_ms\":"<<(p95?*p95:-1.0)<<","
        <<"\"ttft_ms\":"<<(ttft?*ttft:-1.0)<<","
        <<"\"ttfa_ms\":"<<(ttfa?*ttfa:-1.0)<<","
        <<"\"first_audio_chunk_frames\":"
        <<(first_audio_chunk_frames?*first_audio_chunk_frames:-1)<<","
        <<"\"first_audio_chunk_duration_ms\":"
        <<(first_audio_chunk_frames?*first_audio_chunk_frames*80.0:-1.0)<<","
        <<"\"streaming_chunk_frames\":"
        <<(streaming_chunk_frames?*streaming_chunk_frames:-1)<<","
        <<"\"generation_ms\":"<<(generation?*generation:-1.0)<<","
        <<"\"decoder_ms\":"<<(decoder?*decoder:-1.0)<<","
        <<"\"e2e_ms\":"<<(e2e?*e2e:-1.0)<<","
        <<"\"realtime_x\":"<<(realtime?*realtime:-1.0)<<","
        <<"\"generation_wgp\":28,"
        <<"\"decoder_wgp\":20,"
        <<"\"output\":\""
        <<json_escape(
            fs::absolute(
                request.output_wav)
                .string())
        <<"\""
        <<"}\n"
        <<std::flush;
}

static int run_once(
    int argc,
    char** argv) {

    std::uint64_t seed=1234;
    std::string output="qwen3_tts.wav";
    std::string task;
    std::string text_mode;
    std::string speaker;
    std::size_t max_new_tokens=0;

    std::vector<char*> forwarded;
    forwarded.reserve(
        static_cast<std::size_t>(argc));

    forwarded.push_back(argv[0]);

    for(int i=1;i<argc;++i){
        const std::string arg=argv[i];

        if(arg=="--lifecycle"){
            ++i;
            continue;
        }

        if(
            arg=="--task"
            &&i+1<argc
        ){
            task=argv[i+1];
        }

        if(
            arg=="--text-mode"
            &&i+1<argc
        ){
            text_mode=argv[i+1];
        }

        if(
            arg=="--speaker"
            &&i+1<argc
        ){
            speaker=argv[i+1];
        }

        if(
            arg=="--max-new-tokens"
            &&i+1<argc
        ){
            max_new_tokens=
                static_cast<std::size_t>(
                    std::stoull(
                        argv[i+1]));
        }

        if(
            arg=="--seed"
            &&i+1<argc
        ){
            seed=
                static_cast<std::uint64_t>(
                    std::stoull(
                        argv[i+1]));
        }

        if(
            arg=="--output"
            &&i+1<argc
        ){
            output=argv[i+1];
        }

        forwarded.push_back(argv[i]);
    }

    if(
        task!="base-xvector"
        &&task!="custom-voice"
        &&task!="voice-design"
    ){
        throw std::runtime_error(
            "--task must be explicitly base-xvector, custom-voice, or voice-design");
    }

    if(text_mode!="streaming"){
        throw std::runtime_error(
            "--text-mode streaming is required");
    }

    CapturedIo capture;
    capture.begin();

    const int rc=
        qingming_qwen3_tts_rx7900xtx_24g_reference_main(
            static_cast<int>(
                forwarded.size()),
            forwarded.data());

    capture.end();

    print_once_result(
        capture.out.str(),
        capture.err.str(),
        fs::absolute(
            output)
            .string(),
        task,
        text_mode,
        speaker,
        seed,
        max_new_tokens,
        rc);

    return rc;
}

static int run_resident(
    int argc,
    char** argv) {

    fs::path model_dir;
    std::string task;
    std::string text_mode;
    std::size_t maximum_frames=0;

    for(int i=1;i<argc;++i){
        const std::string arg=argv[i];

        auto require_value=
            [&](const char* name)
            ->std::string {

                if(i+1>=argc){
                    throw std::runtime_error(
                        std::string(name)
                        +" requires a value");
                }

                return argv[++i];
            };

        if(arg=="--lifecycle"){
            const auto value=
                require_value(
                    "--lifecycle");

            if(value!="resident"){
                throw std::runtime_error(
                    "--lifecycle must be resident");
            }
        }else if(arg=="--task"){
            task=
                require_value(
                    "--task");
        }else if(arg=="--text-mode"){
            text_mode=
                require_value(
                    "--text-mode");
        }else if(arg=="--model-dir"){
            model_dir=
                fs::path(
                    require_value(
                        "--model-dir"));
        }else if(arg=="--max-new-tokens"){
            const unsigned long long value=
                std::stoull(
                    require_value(
                        "--max-new-tokens"));

            if(value==0||value>8192){
                throw std::runtime_error(
                    "--max-new-tokens must be in [1,8192]");
            }
            std::size_t capacity=256;
            while(capacity<value)capacity<<=1;
            maximum_frames=capacity;
        }else if(
            arg=="--help"
            ||arg=="-h"
        ){
            std::cout
                <<"Usage:\n"
                <<"  "<<argv[0]
                <<" --lifecycle resident"
                <<" --model-dir DIR"
                <<" --max-new-tokens N\n"
                <<"\n"
                <<"Input: one JSON request per line.\n"
                <<"Shutdown: {\"command\":\"shutdown\"}\n";
            return 0;
        }else{
            throw std::runtime_error(
                "unknown resident startup argument: "
                +arg);
        }
    }

    if(maximum_frames==0){
        throw std::runtime_error(
            "--max-new-tokens is required for resident lifecycle");
    }

    if(
        task!="base-xvector"
        &&task!="custom-voice"
        &&task!="voice-design"
    ){
        throw std::runtime_error(
            "--task must be explicitly base-xvector, custom-voice, or voice-design");
    }

    if(text_mode!="streaming"){
        throw std::runtime_error(
            "--text-mode streaming is required");
    }

    if(model_dir.empty()){
        throw std::runtime_error(
            "--model-dir is required");
    }

    // RX7900XTX-24G resident execution partition.
    qwen3_tts::baseline::
        g_resident_generation_wgps=28;

    qwen3_tts::baseline::
        g_resident_decoder_wgps=20;

    qwen3_tts::baseline::
        g_resident_cu_partition_enabled=true;

    std::unique_ptr<
        qwen3_tts::baseline::ResidentEngine>
        engine;

    {
        CapturedIo capture;
        capture.begin();

        try{
            engine=
                std::make_unique<
                    qwen3_tts::baseline::
                        ResidentEngine>(
                            model_dir,
                            maximum_frames,
                            task,
                            text_mode);
        }catch(...){
            capture.end();
            throw;
        }

        capture.end();
    }

    std::cout
        <<"> "
        <<std::flush;

    std::string line;
    std::size_t request_index=0;

    while(std::getline(std::cin,line)){
        const auto begin=
            line.find_first_not_of(
                " \t\r\n");

        if(begin==std::string::npos){
            std::cout<<"> "<<std::flush;
            continue;
        }

        if(
            line=="quit"
            ||line=="exit"
        ){
            break;
        }

        try{
            if(
                const auto command=
                    qwen3_tts::baseline::
                        resident_json_string_field(
                            line,
                            "command")
            ){
                if(*command=="shutdown"){
                    std::cout.flush();
                    std::cerr.flush();

#if defined(__linux__)
                    ::_exit(0);
#else
                    std::_Exit(0);
#endif
                }

                throw std::runtime_error(
                    "unknown command: "
                    +*command);
            }

            auto request=
                qwen3_tts::baseline::
                    parse_resident_request(
                        line,
                        maximum_frames);

            ++request_index;

            CapturedIo capture;
            capture.begin();

            try{
                engine->generate(
                    request,
                    request_index);
            }catch(...){
                capture.end();
                throw;
            }

            capture.end();

            print_resident_result(
                capture.out.str(),
                request,
                task,
                text_mode);

        }catch(const std::exception& error){
            print_error_result(
                "resident",
                error.what());
        }

        std::cout
            <<"> "
            <<std::flush;
    }

    return 0;
}

} // namespace qingming::production

int main(int argc, char** argv) {
    std::string lifecycle="once";

    for(int i=1;i<argc;++i){
        if(std::string(argv[i])=="--lifecycle"){
            if(i+1>=argc){
                qingming::production::
                    print_error_result(
                        "unknown",
                        "--lifecycle requires once or resident");
                return 1;
            }

            lifecycle=argv[i+1];
            break;
        }
    }

    try{
        if(lifecycle=="resident"){
            return qingming::production::
                run_resident(
                    argc,
                    argv);
        }

        if(lifecycle=="once"){
            return qingming::production::
                run_once(
                    argc,
                    argv);
        }

        qingming::production::
            print_error_result(
                lifecycle,
                "--lifecycle must be once or resident");

        return 1;

    }catch(const std::exception& error){
        qingming::production::
            print_error_result(
                lifecycle,
                error.what());

        return 1;
    }
}



#elif defined(QINGMING_MODEL_FAMILY_0_6B)

#define main qingming_qwen3_tts_rx7900xtx_24g_0_6b_reference_main
#include "devices/rx7900xtx-24g/qwen3_tts_0_6b.cpp"
#undef main

namespace qingming::production {

struct CapturedIo {
    std::ostringstream out;
    std::ostringstream err;
    std::streambuf* old_out=nullptr;
    std::streambuf* old_err=nullptr;

    void begin() {
        old_out=std::cout.rdbuf(out.rdbuf());
        old_err=std::cerr.rdbuf(err.rdbuf());
    }

    void end() {
        if(old_out!=nullptr){
            std::cout.rdbuf(old_out);
            old_out=nullptr;
        }

        if(old_err!=nullptr){
            std::cerr.rdbuf(old_err);
            old_err=nullptr;
        }
    }

    ~CapturedIo() {
        end();
    }
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

static std::optional<double> find_number(
    const std::string& text,
    const std::string& key) {

    const std::string needle=key+":";
    const std::size_t begin=text.rfind(needle);

    if(begin==std::string::npos){
        return std::nullopt;
    }

    std::size_t position=begin+needle.size();

    while(
        position<text.size()
        &&std::isspace(
            static_cast<unsigned char>(
                text[position]))
    ){
        ++position;
    }

    const char* first=text.c_str()+position;
    char* end=nullptr;
    const double value=std::strtod(first,&end);

    if(end==first){
        return std::nullopt;
    }

    return value;
}

static std::optional<long long> find_integer(
    const std::string& text,
    const std::string& key) {

    const auto value=
        find_number(
            text,
            key);

    if(!value){
        return std::nullopt;
    }

    return static_cast<long long>(*value);
}

static std::optional<std::string> find_scalar_text(
    const std::string& text,
    const std::string& key) {

    const std::string needle=key+":";
    const std::size_t begin=text.rfind(needle);
    if(begin==std::string::npos){
        return std::nullopt;
    }

    std::size_t position=begin+needle.size();
    while(
        position<text.size()
        &&std::isspace(
            static_cast<unsigned char>(text[position]))
    ){
        ++position;
    }

    const std::size_t start=position;
    while(
        position<text.size()
        &&!std::isspace(
            static_cast<unsigned char>(text[position]))
    ){
        ++position;
    }

    if(position==start){
        return std::nullopt;
    }
    return text.substr(start,position-start);
}

static bool find_eos(
    const std::string& text) {

    return
        text.find(
            "legacy_native_recurrence_stop_c: reason=eos frame=")
        !=std::string::npos;
}

static std::optional<long long> find_eos_frame(
    const std::string& text) {

    const std::string needle=
        "legacy_native_recurrence_stop_c: reason=eos frame=";

    const std::size_t position=
        text.rfind(needle);

    if(position==std::string::npos){
        return std::nullopt;
    }

    const char* first=
        text.c_str()
        +position
        +needle.size();

    char* end=nullptr;

    const long long frame=
        std::strtoll(
            first,
            &end,
            10);

    if(end==first){
        return std::nullopt;
    }

    return frame;
}

static void print_error_result(
    const std::string& lifecycle,
    const std::string& message) {

    std::cout
        <<"{"
        <<"\"status\":\"error\","
        <<"\"device\":\"rx7900xtx-24g\","
        <<"\"family\":\"0.6b\","
        <<"\"execution_profile\":\"0.6b-customvoice-explicit-v2-first8\","
        <<"\"lifecycle\":\""
        <<json_escape(lifecycle)
        <<"\","
        <<"\"error\":\""
        <<json_escape(message)
        <<"\""
        <<"}\n"
        <<std::flush;
}

static void print_once_result(
    const std::string& captured,
    const std::string& captured_error,
    const std::string& output,
    const std::string& task,
    const std::string& text_mode,
    const std::string& speaker,
    std::uint64_t seed,
    std::size_t max_new_tokens,
    int rc) {

    const auto frames=
        find_integer(
            captured,
            "legacy_codec_frame_count_b");

    const auto codec_hash=
        find_scalar_text(
            captured,
            "legacy_codec_fnv1a64_b");

    const auto duration=
        find_number(
            captured,
            "legacy_audio_duration_s_b");

    const auto ttft=
        find_number(
            captured,
            "legacy_ttft_ms_b");

    const auto ttfa=
        find_number(
            captured,
            "legacy_ttfa_ms_b");

    const auto first_audio_chunk_frames=
        find_integer(
            captured,
            "streaming_decoder_first_chunk_frames");

    const auto streaming_chunk_frames=
        find_integer(
            captured,
            "streaming_decoder_chunk_frames");

    const auto generation=
        find_number(
            captured,
            "legacy_codec_generation_ms_b");

    const auto decoder=
        find_number(
            captured,
            "legacy_codec_decoder_graph_ms_b");

    const auto e2e=
        find_number(
            captured,
            "legacy_e2e_ms_b");

    const auto realtime=
        find_number(
            captured,
            "legacy_e2e_realtime_x_b");

    const auto p50=
        find_number(
            captured,
            "predictor_frame_p50_ms");

    const auto p95=
        find_number(
            captured,
            "predictor_frame_p95_ms");

    const bool eos=find_eos(captured);
    const auto eos_frame=find_eos_frame(captured);

    const bool ok=
        rc==0
        &&frames.has_value()
        &&e2e.has_value();

    std::cout
        <<std::fixed
        <<std::setprecision(3)
        <<"{"
        <<"\"status\":\""
        <<(ok?"ok":"error")
        <<"\","
        <<"\"device\":\"rx7900xtx-24g\","
        <<"\"family\":\"0.6b\","
        <<"\"execution_profile\":\"0.6b-customvoice-explicit-v2-first8\","
        <<"\"lifecycle\":\"once\","
        <<"\"task\":\""<<json_escape(task)<<"\","
        <<"\"text_mode\":\""<<json_escape(text_mode)<<"\","
        <<"\"speaker\":\""<<json_escape(speaker)<<"\","
        <<"\"error\":\""
        <<json_escape(
            ok
            ?std::string{}
            :captured_error)
        <<"\","
        <<"\"seed\":"<<seed<<","
        <<"\"max_new_tokens\":"<<max_new_tokens<<","
        <<"\"max_new_tokens_capacity\":"
        <<([&](){
            if(max_new_tokens==0||max_new_tokens>8192)return std::size_t{0};
            std::size_t capacity=256;
            while(capacity<max_new_tokens)capacity<<=1;
            return capacity;
        })()<<","
        <<"\"frames\":"<<(frames?*frames:-1)<<","
        <<"\"codec_fnv1a64\":\""
        <<json_escape(codec_hash?*codec_hash:std::string{})
        <<"\","
        <<"\"eos\":"<<(eos?"true":"false")<<","
        <<"\"eos_frame\":"<<(eos_frame?*eos_frame:-1)<<","
        <<"\"audio_duration_s\":"<<(duration?*duration:-1.0)<<","
        <<"\"predictor_p50_ms\":"<<(p50?*p50:-1.0)<<","
        <<"\"predictor_p95_ms\":"<<(p95?*p95:-1.0)<<","
        <<"\"ttft_ms\":"<<(ttft?*ttft:-1.0)<<","
        <<"\"ttfa_ms\":"<<(ttfa?*ttfa:-1.0)<<","
        <<"\"first_audio_chunk_frames\":"
        <<(first_audio_chunk_frames?*first_audio_chunk_frames:-1)<<","
        <<"\"first_audio_chunk_duration_ms\":"
        <<(first_audio_chunk_frames?*first_audio_chunk_frames*80.0:-1.0)<<","
        <<"\"streaming_chunk_frames\":"
        <<(streaming_chunk_frames?*streaming_chunk_frames:-1)<<","
        <<"\"generation_ms\":"<<(generation?*generation:-1.0)<<","
        <<"\"decoder_ms\":"<<(decoder?*decoder:-1.0)<<","
        <<"\"e2e_ms\":"<<(e2e?*e2e:-1.0)<<","
        <<"\"realtime_x\":"<<(realtime?*realtime:-1.0)<<","
        <<"\"output\":\""<<json_escape(output)<<"\""
        <<"}\n"
        <<std::flush;
}

static void print_resident_result(
    const std::string& captured,
    const qwen3_tts::baseline::ResidentRequest& request,
    const std::string& task,
    const std::string& text_mode) {

    const auto frames=
        find_integer(
            captured,
            "resident_codec_frame_count");

    const auto codec_hash=
        find_scalar_text(
            captured,
            "resident_codec_fnv1a64");

    const auto duration=
        find_number(
            captured,
            "resident_audio_duration_s");

    const auto ttft=
        find_number(
            captured,
            "resident_ttft_ms");

    const auto ttfa=
        find_number(
            captured,
            "resident_ttfa_ms");

    const auto first_audio_chunk_frames=
        find_integer(
            captured,
            "resident_streaming_decoder_first_chunk_frames");

    const auto streaming_chunk_frames=
        find_integer(
            captured,
            "resident_streaming_decoder_chunk_frames");

    const auto generation=
        find_number(
            captured,
            "resident_generation_ms");

    const auto decoder=
        find_number(
            captured,
            "resident_decoder_compute_ms");

    const auto e2e=
        find_number(
            captured,
            "resident_e2e_ms");

    const auto realtime=
        find_number(
            captured,
            "resident_e2e_realtime_x");

    const auto p50=
        find_number(
            captured,
            "resident_predictor_frame_p50_ms");

    const auto p95=
        find_number(
            captured,
            "resident_predictor_frame_p95_ms");

    const bool eos=find_eos(captured);
    const auto eos_frame=find_eos_frame(captured);

    const bool ok=
        frames.has_value()
        &&e2e.has_value()
        &&captured.find(
            "resident_request_success: True")
            !=std::string::npos;

    std::cout
        <<std::fixed
        <<std::setprecision(3)
        <<"{"
        <<"\"status\":\""
        <<(ok?"ok":"error")
        <<"\","
        <<"\"device\":\"rx7900xtx-24g\","
        <<"\"family\":\"0.6b\","
        <<"\"execution_profile\":\"0.6b-customvoice-explicit-v2-first8\","
        <<"\"lifecycle\":\"resident\","
        <<"\"task\":\""<<json_escape(task)<<"\","
        <<"\"text_mode\":\""<<json_escape(text_mode)<<"\","
        <<"\"speaker\":\""<<json_escape(request.speaker)<<"\","
        <<"\"seed\":"<<request.seed<<","
        <<"\"max_new_tokens\":"<<request.max_new_tokens<<","
        <<"\"max_new_tokens_capacity\":"
        <<([&](){
            if(request.max_new_tokens==0||request.max_new_tokens>8192)return std::size_t{0};
            std::size_t capacity=256;
            while(capacity<request.max_new_tokens)capacity<<=1;
            return capacity;
        })()<<","
        <<"\"frames\":"<<(frames?*frames:-1)<<","
        <<"\"codec_fnv1a64\":\""
        <<json_escape(codec_hash?*codec_hash:std::string{})
        <<"\","
        <<"\"eos\":"<<(eos?"true":"false")<<","
        <<"\"eos_frame\":"<<(eos_frame?*eos_frame:-1)<<","
        <<"\"audio_duration_s\":"<<(duration?*duration:-1.0)<<","
        <<"\"predictor_p50_ms\":"<<(p50?*p50:-1.0)<<","
        <<"\"predictor_p95_ms\":"<<(p95?*p95:-1.0)<<","
        <<"\"ttft_ms\":"<<(ttft?*ttft:-1.0)<<","
        <<"\"ttfa_ms\":"<<(ttfa?*ttfa:-1.0)<<","
        <<"\"first_audio_chunk_frames\":"
        <<(first_audio_chunk_frames?*first_audio_chunk_frames:-1)<<","
        <<"\"first_audio_chunk_duration_ms\":"
        <<(first_audio_chunk_frames?*first_audio_chunk_frames*80.0:-1.0)<<","
        <<"\"streaming_chunk_frames\":"
        <<(streaming_chunk_frames?*streaming_chunk_frames:-1)<<","
        <<"\"generation_ms\":"<<(generation?*generation:-1.0)<<","
        <<"\"decoder_ms\":"<<(decoder?*decoder:-1.0)<<","
        <<"\"e2e_ms\":"<<(e2e?*e2e:-1.0)<<","
        <<"\"realtime_x\":"<<(realtime?*realtime:-1.0)<<","
        <<"\"generation_wgp\":28,"
        <<"\"decoder_wgp\":20,"
        <<"\"output\":\""
        <<json_escape(
            fs::absolute(
                request.output_wav)
                .string())
        <<"\""
        <<"}\n"
        <<std::flush;
}

static int run_once(
    int argc,
    char** argv) {

    std::uint64_t seed=1234;
    std::string output="qwen3_tts.wav";
    std::string task;
    std::string text_mode;
    std::string speaker;
    std::size_t max_new_tokens=0;

    std::vector<char*> forwarded;
    forwarded.reserve(
        static_cast<std::size_t>(argc));

    forwarded.push_back(argv[0]);

    for(int i=1;i<argc;++i){
        const std::string arg=argv[i];

        if(arg=="--lifecycle"){
            ++i;
            continue;
        }

        if(
            arg=="--task"
            &&i+1<argc
        ){
            task=argv[i+1];
        }

        if(
            arg=="--text-mode"
            &&i+1<argc
        ){
            text_mode=argv[i+1];
        }

        if(
            arg=="--speaker"
            &&i+1<argc
        ){
            speaker=argv[i+1];
        }

        if(
            arg=="--max-new-tokens"
            &&i+1<argc
        ){
            max_new_tokens=
                static_cast<std::size_t>(
                    std::stoull(
                        argv[i+1]));
        }

        if(
            arg=="--seed"
            &&i+1<argc
        ){
            seed=
                static_cast<std::uint64_t>(
                    std::stoull(
                        argv[i+1]));
        }

        if(
            arg=="--output"
            &&i+1<argc
        ){
            output=argv[i+1];
        }

        forwarded.push_back(argv[i]);
    }

    if(
        task!="base-xvector"
        &&task!="custom-voice"
    ){
        throw std::runtime_error(
            "--task must be explicitly base-xvector or custom-voice");
    }

    if(text_mode!="streaming"){
        throw std::runtime_error(
            "--text-mode streaming is required");
    }

    CapturedIo capture;
    capture.begin();

    const int rc=
        qingming_qwen3_tts_rx7900xtx_24g_0_6b_reference_main(
            static_cast<int>(
                forwarded.size()),
            forwarded.data());

    capture.end();

    print_once_result(
        capture.out.str(),
        capture.err.str(),
        fs::absolute(
            output)
            .string(),
        task,
        text_mode,
        speaker,
        seed,
        max_new_tokens,
        rc);

    return rc;
}

static int run_resident(
    int argc,
    char** argv) {

    fs::path model_dir;
    std::string task;
    std::string text_mode;
    std::size_t maximum_frames=0;

    for(int i=1;i<argc;++i){
        const std::string arg=argv[i];

        auto require_value=
            [&](const char* name)
            ->std::string {

                if(i+1>=argc){
                    throw std::runtime_error(
                        std::string(name)
                        +" requires a value");
                }

                return argv[++i];
            };

        if(arg=="--lifecycle"){
            const auto value=
                require_value(
                    "--lifecycle");

            if(value!="resident"){
                throw std::runtime_error(
                    "--lifecycle must be resident");
            }
        }else if(arg=="--task"){
            task=
                require_value(
                    "--task");
        }else if(arg=="--text-mode"){
            text_mode=
                require_value(
                    "--text-mode");
        }else if(arg=="--model-dir"){
            model_dir=
                fs::path(
                    require_value(
                        "--model-dir"));
        }else if(arg=="--max-new-tokens"){
            const unsigned long long value=
                std::stoull(
                    require_value(
                        "--max-new-tokens"));

            if(value==0||value>8192){
                throw std::runtime_error(
                    "--max-new-tokens must be in [1,8192]");
            }
            std::size_t capacity=256;
            while(capacity<value)capacity<<=1;
            maximum_frames=capacity;
        }else if(
            arg=="--help"
            ||arg=="-h"
        ){
            std::cout
                <<"Usage:\n"
                <<"  "<<argv[0]
                <<" --lifecycle resident"
                <<" --model-dir DIR"
                <<" --max-new-tokens N\n"
                <<"\n"
                <<"Input: one JSON request per line.\n"
                <<"Shutdown: {\"command\":\"shutdown\"}\n";
            return 0;
        }else{
            throw std::runtime_error(
                "unknown resident startup argument: "
                +arg);
        }
    }

    if(maximum_frames==0){
        throw std::runtime_error(
            "--max-new-tokens is required for resident lifecycle");
    }

    if(
        task!="base-xvector"
        &&task!="custom-voice"
    ){
        throw std::runtime_error(
            "--task must be explicitly base-xvector or custom-voice");
    }

    if(text_mode!="streaming"){
        throw std::runtime_error(
            "--text-mode streaming is required");
    }

    if(model_dir.empty()){
        throw std::runtime_error(
            "--model-dir is required");
    }

    
    qwen3_tts::baseline::
        g_resident_generation_wgps=28;

    qwen3_tts::baseline::
        g_resident_decoder_wgps=20;

    qwen3_tts::baseline::
        g_resident_cu_partition_enabled=true;

    std::unique_ptr<
        qwen3_tts::baseline::ResidentEngine>
        engine;

    {
        CapturedIo capture;
        capture.begin();

        try{
            engine=
                std::make_unique<
                    qwen3_tts::baseline::
                        ResidentEngine>(
                            model_dir,
                            maximum_frames,
                            task,
                            text_mode);
        }catch(...){
            capture.end();
            throw;
        }

        capture.end();
    }

    std::cout
        <<"> "
        <<std::flush;

    std::string line;
    std::size_t request_index=0;

    while(std::getline(std::cin,line)){
        const auto begin=
            line.find_first_not_of(
                " \t\r\n");

        if(begin==std::string::npos){
            std::cout<<"> "<<std::flush;
            continue;
        }

        if(
            line=="quit"
            ||line=="exit"
        ){
            break;
        }

        try{
            if(
                const auto command=
                    qwen3_tts::baseline::
                        resident_json_string_field(
                            line,
                            "command")
            ){
                if(*command=="shutdown"){
                    std::cout.flush();
                    std::cerr.flush();

#if defined(__linux__)
                    ::_exit(0);
#else
                    std::_Exit(0);
#endif
                }

                throw std::runtime_error(
                    "unknown command: "
                    +*command);
            }

            auto request=
                qwen3_tts::baseline::
                    parse_resident_request(
                        line,
                        maximum_frames);

            ++request_index;

            CapturedIo capture;
            capture.begin();

            try{
                engine->generate(
                    request,
                    request_index);
            }catch(...){
                capture.end();
                throw;
            }

            capture.end();

            print_resident_result(
                capture.out.str(),
                request,
                task,
                text_mode);

        }catch(const std::exception& error){
            print_error_result(
                "resident",
                error.what());
        }

        std::cout
            <<"> "
            <<std::flush;
    }

    return 0;
}

} // namespace qingming::production

int main(int argc, char** argv) {
    std::string lifecycle="once";

    for(int i=1;i<argc;++i){
        if(std::string(argv[i])=="--lifecycle"){
            if(i+1>=argc){
                qingming::production::
                    print_error_result(
                        "unknown",
                        "--lifecycle requires once or resident");
                return 1;
            }

            lifecycle=argv[i+1];
            break;
        }
    }

    try{
        if(lifecycle=="resident"){
            return qingming::production::
                run_resident(
                    argc,
                    argv);
        }

        if(lifecycle=="once"){
            return qingming::production::
                run_once(
                    argc,
                    argv);
        }

        qingming::production::
            print_error_result(
                lifecycle,
                "--lifecycle must be once or resident");

        return 1;

    }catch(const std::exception& error){
        qingming::production::
            print_error_result(
                lifecycle,
                error.what());

        return 1;
    }
}


#else
#error "Exactly one QINGMING_MODEL_FAMILY_* compile definition is required."
#endif
