#include "models.h"

void llama_model_zamba2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 38: type = LLM_TYPE_1_2B; break; // zamba2-1.2B
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_zamba2::load_arch_tensors(llama_model_loader &) {

}

std::unique_ptr<llm_graph_context> llama_model_zamba2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

