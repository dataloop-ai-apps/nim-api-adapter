# Project Metadata


---

# Test Map

Test files: 20 | Source files: 16

## Test files (20)
  agent/tests/local_agent_debug.py
  agent/tests/test_agent.py  -> agent/src/agent.py
  models/downloadable/tests/test_simple.py  -> models/downloadable/src/simple.py
  test/__init__.py
  test/assets/unittests/prompt_item.json
  test/unittests/__init__.py
  test/unittests/test_baidu_paddleocr.py  -> src/unittests/baidu_paddleocr.py
  test/unittests/test_deepseek_ai_deepseek_r1.py  -> src/unittests/deepseek_ai_deepseek_r1.py
  test/unittests/test_google_gemma_7b.py  -> src/unittests/google_gemma_7b.py
  test/unittests/test_ibm_granite_34b_code_instruct.py  -> src/unittests/ibm_granite_34b_code_instruct.py
  test/unittests/test_meta_llama3_70b_instruct.py  -> src/unittests/meta_llama3_70b_instruct.py
  test/unittests/test_meta_llama3_8b_instruct.py  -> src/unittests/meta_llama3_8b_instruct.py
  test/unittests/test_meta_llama_3_1_70b_instruct.py  -> src/unittests/meta_llama_3_1_70b_instruct.py
  test/unittests/test_meta_llama_3_2_90b_vision_instruct.py  -> src/unittests/meta_llama_3_2_90b_vision_instruct.py
  test/unittests/test_meta_llama_3_3_70b_instruct.py  -> src/unittests/meta_llama_3_3_70b_instruct.py
  test/unittests/test_microsoft_kosmos_2.py  -> src/unittests/microsoft_kosmos_2.py
  test/unittests/test_mistralai_mistral_large.py  -> src/unittests/mistralai_mistral_large.py
  test/unittests/test_nv_yolox_page_elements_v1.py  -> src/unittests/nv_yolox_page_elements_v1.py
  test/unittests/test_nvidia_neva_22b.py  -> src/unittests/nvidia_neva_22b.py
  test/unittests/test_university_at_buffalo_cached.py  -> src/unittests/university_at_buffalo_cached.py


---

# Documentation Index

## Root
  README.md: "NVIDIA NIM Adapter for Dataloop"
  support_matrix.md: "NIM Adapter Support Matrix"

## .plans
  .plans/CODE_GRAPH.md: "Code Structure Index"

## agent
  agent/README.md: "NVIDIA NIM Agent"

## models/downloadable
  models/downloadable/README.md: "Downloadable NIM Models for Dataloop"


---

# Config Files

  .github/workflows/nim-agent.yml                                                              (CI pipeline)
  Dockerfile                                                                                   (container)
  models/api/embeddings/baai/bge_m3/dataloop.json                                              (dataloop)
  models/api/embeddings/nvidia/llama_3_2_nemoretriever_1b_vlm_embed_v1/dataloop.json           (dataloop)
  models/api/embeddings/nvidia/llama_3_2_nemoretriever_300m_embed_v1/dataloop.json             (dataloop)
  models/api/embeddings/nvidia/llama_3_2_nemoretriever_300m_embed_v2/dataloop.json             (dataloop)
  models/api/embeddings/nvidia/llama_3_2_nv_embedqa_1b_v2/dataloop.json                        (dataloop)
  models/api/embeddings/nvidia/nv_embed_v1/dataloop.json                                       (dataloop)
  models/api/embeddings/nvidia/nv_embedcode_7b_v1/dataloop.json                                (dataloop)
  models/api/embeddings/nvidia/nv_embedqa_e5_v5/dataloop.json                                  (dataloop)
  models/api/llm/abacusai/dracarys_llama_3_1_70b_instruct/dataloop.json                        (dataloop)
  models/api/llm/ai21labs/jamba_1_5_mini_instruct/dataloop.json                                (dataloop)
  models/api/llm/baichuan_inc/baichuan2_13b_chat/dataloop.json                                 (dataloop)
  models/api/llm/bytedance/seed_oss_36b_instruct/dataloop.json                                 (dataloop)
  models/api/llm/google/gemma_3_12b_it/dataloop.json                                           (dataloop)
  models/api/llm/google/gemma_3_1b_it/dataloop.json                                            (dataloop)
  models/api/llm/google/gemma_3_27b_it/dataloop.json                                           (dataloop)
  models/api/llm/google/gemma_3_4b_it/dataloop.json                                            (dataloop)
  models/api/llm/google/gemma_3n_e2b_it/dataloop.json                                          (dataloop)
  models/api/llm/google/gemma_3n_e4b_it/dataloop.json                                          (dataloop)
  models/api/llm/gotocompany/gemma_2_9b_cpt_sahabatai_instruct/dataloop.json                   (dataloop)
  models/api/llm/ibm/granite_3_3_8b_instruct/dataloop.json                                     (dataloop)
  models/api/llm/igenius/italia_10b_instruct_16k/dataloop.json                                 (dataloop)
  models/api/llm/institute_of_science_tokyo/llama_3_1_swallow_8b_instruct_v0_1/dataloop.json   (dataloop)
  models/api/llm/meta/llama2_70b/dataloop.json                                                 (dataloop)
  models/api/llm/meta/llama3_70b_instruct/dataloop.json                                        (dataloop)
  models/api/llm/meta/llama3_8b_instruct/dataloop.json                                         (dataloop)
  models/api/llm/meta/llama_3_1_405b_instruct/dataloop.json                                    (dataloop)
  models/api/llm/meta/llama_3_1_70b_instruct/dataloop.json                                     (dataloop)
  models/api/llm/meta/llama_3_1_8b_instruct/dataloop.json                                      (dataloop)
  models/api/llm/meta/llama_3_2_1b_instruct/dataloop.json                                      (dataloop)
  models/api/llm/meta/llama_3_2_3b_instruct/dataloop.json                                      (dataloop)
  models/api/llm/meta/llama_3_3_70b_instruct/dataloop.json                                     (dataloop)
  models/api/llm/meta/llama_4_maverick_17b_128e_instruct/dataloop.json                         (dataloop)
  models/api/llm/meta/llama_4_scout_17b_16e_instruct/dataloop.json                             (dataloop)
  models/api/llm/meta/llama_guard_4_12b/dataloop.json                                          (dataloop)
  models/api/llm/microsoft/phi_3_5_mini_instruct/dataloop.json                                 (dataloop)
  models/api/llm/microsoft/phi_3_5_moe_instruct/dataloop.json                                  (dataloop)
  models/api/llm/microsoft/phi_3_medium_128k_instruct/dataloop.json                            (dataloop)
  models/api/llm/microsoft/phi_3_mini_128k_instruct/dataloop.json                              (dataloop)
  models/api/llm/microsoft/phi_4_mini_flash_reasoning/dataloop.json                            (dataloop)
  models/api/llm/microsoft/phi_4_mini_instruct/dataloop.json                                   (dataloop)
  models/api/llm/minimaxai/minimax_m2/dataloop.json                                            (dataloop)
  models/api/llm/minimaxai/minimax_m2_1/dataloop.json                                          (dataloop)
  models/api/llm/mistralai/magistral_small_2506/dataloop.json                                  (dataloop)
  models/api/llm/mistralai/mamba_codestral_7b_v0_1/dataloop.json                               (dataloop)
  models/api/llm/mistralai/mathstral_7b_v0_1/dataloop.json                                     (dataloop)
  models/api/llm/mistralai/mistral_7b_instruct_v0_2/dataloop.json                              (dataloop)
  models/api/llm/mistralai/mistral_medium_3_instruct/dataloop.json                             (dataloop)
  models/api/llm/mistralai/mistral_nemotron/dataloop.json                                      (dataloop)
  models/api/llm/mistralai/mistral_small_24b_instruct/dataloop.json                            (dataloop)
  models/api/llm/mistralai/mistral_small_3_1_24b_instruct_2503/dataloop.json                   (dataloop)
  models/api/llm/mistralai/mixtral_8x22b_instruct_v0_1/dataloop.json                           (dataloop)
  models/api/llm/mistralai/mixtral_8x7b_instruct_v0_1/dataloop.json                            (dataloop)
  models/api/llm/moonshotai/kimi_k2_5/dataloop.json                                            (dataloop)
  models/api/llm/moonshotai/kimi_k2_instruct/dataloop.json                                     (dataloop)
  models/api/llm/moonshotai/kimi_k2_instruct_0905/dataloop.json                                (dataloop)
  models/api/llm/nvidia/llama_3_1_nemoguard_8b_content_safety/dataloop.json                    (dataloop)
  models/api/llm/nvidia/llama_3_1_nemoguard_8b_topic_control/dataloop.json                     (dataloop)
  models/api/llm/nvidia/llama_3_1_nemotron_nano_4b_v1_1/dataloop.json                          (dataloop)
  models/api/llm/nvidia/llama_3_1_nemotron_nano_8b_v1/dataloop.json                            (dataloop)
  models/api/llm/nvidia/llama_3_1_nemotron_nano_vl_8b_v1/dataloop.json                         (dataloop)
  models/api/llm/nvidia/llama_3_1_nemotron_safety_guard_8b_v3/dataloop.json                    (dataloop)
  models/api/llm/nvidia/llama_3_1_nemotron_ultra_253b_v1/dataloop.json                         (dataloop)
  models/api/llm/nvidia/llama_3_3_nemotron_super_49b_v1_5/dataloop.json                        (dataloop)
  models/api/llm/nvidia/nemotron_3_nano_30b_a3b/dataloop.json                                  (dataloop)
  models/api/llm/nvidia/nemotron_content_safety_reasoning_4b/dataloop.json                     (dataloop)
  models/api/llm/nvidia/nemotron_mini_4b_instruct/dataloop.json                                (dataloop)
  models/api/llm/nvidia/nemotron_nano_12b_v2_vl/dataloop.json                                  (dataloop)
  models/api/llm/nvidia/nvidia_nemotron_nano_9b_v2/dataloop.json                               (dataloop)
  models/api/llm/nvidia/riva_translate_4b_instruct/dataloop.json                               (dataloop)
  models/api/llm/nvidia/riva_translate_4b_instruct_v1_1/dataloop.json                          (dataloop)
  models/api/llm/openai/gpt_oss_20b/dataloop.json                                              (dataloop)
  models/api/llm/qwen/qwen2_5_7b_instruct/dataloop.json                                        (dataloop)
  models/api/llm/qwen/qwen2_5_coder_32b_instruct/dataloop.json                                 (dataloop)
  models/api/llm/qwen/qwen2_5_coder_7b_instruct/dataloop.json                                  (dataloop)
  models/api/llm/qwen/qwen2_7b_instruct/dataloop.json                                          (dataloop)
  models/api/llm/qwen/qwen3_235b_a22b/dataloop.json                                            (dataloop)
  models/api/llm/qwen/qwen3_coder_480b_a35b_instruct/dataloop.json                             (dataloop)
  models/api/llm/qwen/qwen3_next_80b_a3b_instruct/dataloop.json                                (dataloop)
  models/api/llm/qwen/qwen3_next_80b_a3b_thinking/dataloop.json                                (dataloop)
  models/api/llm/sarvamai/sarvam_m/dataloop.json                                               (dataloop)
  models/api/llm/speakleash/bielik_11b_v2_3_instruct/dataloop.json                             (dataloop)
  models/api/llm/speakleash/bielik_11b_v2_6_instruct/dataloop.json                             (dataloop)
  models/api/llm/stepfun_ai/step_3_5_flash/dataloop.json                                       (dataloop)
  models/api/llm/thudm/chatglm3_6b/dataloop.json                                               (dataloop)
  models/api/llm/tiiuae/falcon3_7b_instruct/dataloop.json                                      (dataloop)
  models/api/llm/tokyotech_llm/llama_3_swallow_70b_instruct_v0_1/dataloop.json                 (dataloop)
  models/api/llm/upstage/solar_10_7b_instruct/dataloop.json                                    (dataloop)
  models/api/llm/z_ai/glm4_7/dataloop.json                                                     (dataloop)
  models/api/object_detection/baidu_paddleocr/dataloop.json                                    (dataloop)
  models/api/object_detection/nv_yolox_page_elements_v1/dataloop.json                          (dataloop)
  models/api/object_detection/university_at_buffalo_cached/dataloop.json                       (dataloop)
  models/api/vlm/meta/llama_3_2_11b_vision_instruct/dataloop.json                              (dataloop)
  models/api/vlm/meta/llama_3_2_90b_vision_instruct/dataloop.json                              (dataloop)
  models/api/vlm/microsoft/phi_3_5_vision_instruct/dataloop.json                               (dataloop)
  models/api/vlm/microsoft/phi_4_multimodal_instruct/dataloop.json                             (dataloop)
  models/api/vlm/nvidia/nemotron_nano_12b_v2_vl/dataloop.json                                  (dataloop)
  models/downloadable/embeddings/nvidia/llama_3_2_nemoretriever_300m_embed_v2/dataloop.json    (dataloop)
  models/downloadable/embeddings/nvidia/llama_3_2_nv_embedqa_1b_v2/dataloop.json               (dataloop)
  models/downloadable/embeddings/nvidia/nv_embedqa_e5_v5/dataloop.json                         (dataloop)
  models/downloadable/llm/google/gemma_3_1b_it/dataloop.json                                   (dataloop)
  models/downloadable/llm/meta/llama_3_1_70b_instruct/dataloop.json                            (dataloop)
  models/downloadable/llm/meta/llama_3_1_8b_instruct/dataloop.json                             (dataloop)
  models/downloadable/llm/meta/llama_3_2_1b_instruct/dataloop.json                             (dataloop)
  models/downloadable/llm/meta/llama_3_2_3b_instruct/dataloop.json                             (dataloop)
  models/downloadable/llm/meta/llama_3_3_70b_instruct/dataloop.json                            (dataloop)
  models/downloadable/llm/meta/llama_4_scout_17b_16e_instruct/dataloop.json                    (dataloop)
  models/downloadable/llm/microsoft/phi_4_mini_instruct/dataloop.json                          (dataloop)
  models/downloadable/llm/nvidia/llama_3_1_nemoguard_8b_content_safety/dataloop.json           (dataloop)
  models/downloadable/llm/nvidia/llama_3_1_nemoguard_8b_topic_control/dataloop.json            (dataloop)
  models/downloadable/llm/nvidia/llama_3_1_nemotron_nano_vl_8b_v1/dataloop.json                (dataloop)
  models/downloadable/llm/nvidia/llama_3_1_nemotron_ultra_253b_v1/dataloop.json                (dataloop)
  models/downloadable/llm/nvidia/llama_3_3_nemotron_super_49b_v1_5/dataloop.json               (dataloop)
  models/downloadable/llm/nvidia/nemotron_nano_12b_v2_vl/dataloop.json                         (dataloop)
  models/downloadable/llm/nvidia/nvidia_nemotron_nano_9b_v2/dataloop.json                      (dataloop)
  models/downloadable/llm/openai/gpt_oss_20b/dataloop.json                                     (dataloop)
  models/downloadable/llm/qwen/qwen2_5_coder_32b_instruct/dataloop.json                        (dataloop)
  models/downloadable/llm/qwen/qwen3_next_80b_a3b_instruct/dataloop.json                       (dataloop)
  models/downloadable/llm/qwen/qwen3_next_80b_a3b_thinking/dataloop.json                       (dataloop)
  models/downloadable/llm/sarvamai/sarvam_m/dataloop.json                                      (dataloop)
  models/downloadable/object_detection/baidu_paddleocr/dataloop.json                           (dataloop)
  models/downloadable/vlm/meta/llama_3_2_11b_vision_instruct/dataloop.json                     (dataloop)
  models/downloadable/vlm/meta/llama_3_2_90b_vision_instruct/dataloop.json                     (dataloop)
  nodes/asr/Dockerfile                                                                         (container)
  nodes/asr/nvidia/parakeet_ctc_0_6b_asr/dataloop.json                                         (dataloop)
