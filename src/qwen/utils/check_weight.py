# def check_weight(model):
#     print("--- KIỂM TRA TRỌNG SỐ Encoder RMS NORM ---")
#     encoder = model.get_encoder()
#     for idx, layer in enumerate(encoder.layers):
#         print(f"Layer {idx}")
#         print(
#             f"Input RMS Weight | Max: {layer.input_layernorm.weight.max().item():.4f} "
#             f"| Min: {layer.input_layernorm.weight.min().item():.4f} "
#             f"| Mean: {layer.input_layernorm.weight.mean().item():.4f} "
#             f"| Std: {layer.input_layernorm.weight.std().item():.4f}"
#          )
#         print(
#             f"Post RMS Weight | Max: {layer.post_attention_layernorm.weight.max().item():.4f} "
#             f"| Min: {layer.post_attention_layernorm.weight.min().item():.4f} "
#             f"| Mean: {layer.post_attention_layernorm.weight.mean().item():.4f} "
#             f"| Std: {layer.post_attention_layernorm.weight.std().item():.4f}"
#          )
#     rms_norm = model.get_encoder().norm 
#     print(f"Encoder RMS norm Weight | Max: {rms_norm.weight.max().item():.4f} "
#           f"| Min: {rms_norm.weight.min().item():.4f} "
#           f"| Mean: {rms_norm.weight.mean().item():.4f} "
#           f"| Std: {rms_norm.weight.std().item():.4f}")
    
#     if rms_norm.weight.std().item() > 0.1:
#         print("❌ CẢNH BÁO: Trọng số quá lớn! Hàm khởi tạo chưa chạy đúng.")
#     else:
#         print("✅ Trọng số có vẻ ổn (std nhỏ).")

#     print("--- KIỂM TRA TRỌNG SỐ PROJECT DOWN ---")
#     linear1 = model.get_encoder().connector.encoder_project.linear_1 
#     print(f"Linear 1 Weight | Max: {linear1.weight.max().item():.4f} "
#         f"| Min: {linear1.weight.min().item():.4f} "
#         f"| Mean: {linear1.weight.mean().item():.4f} "
#         f"| Std: {linear1.weight.std().item():.4f}")

#     linear2 = model.get_encoder().connector.encoder_project.linear_2
#     print(f"Linear 2 Weight | Max: {linear2.weight.max().item():.4f} "
#         f"| Min: {linear2.weight.min().item():.4f} "
#         f"| Mean: {linear2.weight.mean().item():.4f} "
#         f"| Std: {linear2.weight.std().item():.4f}")
    
#     if linear1.weight.std().item() > 0.1:
#         print("❌ CẢNH BÁO: Trọng số quá lớn! Hàm khởi tạo chưa chạy đúng.")
#     else:
#         print("✅ Trọng số có vẻ ổn (std nhỏ).")

#     print("--- KIỂM TRA TRỌNG SỐ CONNECTOR ---")
#     for idx, layer in enumerate(encoder.connector.post_encoder.layers):
#         print(f"Layer {idx}")
#         print(
#             f"Shape: {layer.input_layernorm.weight.shape}"
#             f"Input RMS Weight | Max: {layer.input_layernorm.weight.max().item():.4f} "
#             f"| Min: {layer.input_layernorm.weight.min().item():.4f} "
#             f"| Mean: {layer.input_layernorm.weight.mean().item():.4f} "
#             f"| Std: {layer.input_layernorm.weight.std().item():.4f}"
#          )
#         print(
#             f"Shape: {layer.post_attention_layernorm.weight.shape}"
#             f"Post RMS Weight | Max: {layer.post_attention_layernorm.weight.max().item():.4f} "
#             f"| Min: {layer.post_attention_layernorm.weight.min().item():.4f} "
#             f"| Mean: {layer.post_attention_layernorm.weight.mean().item():.4f} "
#             f"| Std: {layer.post_attention_layernorm.weight.std().item():.4f}"
#          )

#     print("--- KIỂM TRA TRỌNG SỐ DECODER ---")
#     decoder = model.get_decoder()
#     for idx, layer in enumerate(decoder.layers):
#         print(f"Layer {idx}")
#         print(
#             f"Shape: {layer.self_attn_layer_norm.weight.shape}"
#             f"Input RMS Weight | Max: {layer.self_attn_layer_norm.weight.max().item():.4f} "
#             f"| Min: {layer.self_attn_layer_norm.weight.min().item():.4f} "
#             f"| Mean: {layer.self_attn_layer_norm.weight.mean().item():.4f} "
#             f"| Std: {layer.self_attn_layer_norm.weight.std().item():.4f}"
#          )
#         print(
#             f"Shape: {layer.encoder_attn_layer_norm.weight.shape}"
#             f"Post RMS Weight | Max: {layer.encoder_attn_layer_norm.weight.max().item():.4f} "
#             f"| Min: {layer.encoder_attn_layer_norm.weight.min().item():.4f} "
#             f"| Mean: {layer.encoder_attn_layer_norm.weight.mean().item():.4f} "
#             f"| Std: {layer.encoder_attn_layer_norm.weight.std().item():.4f}")
    
#     print("--- KIỂM TRA TRỌNG SỐ LM_HEAD ---")
#     lm_head = model.mt_model.lm_head
#     print(lm_head.weight.device)
#     print(lm_head.weight.is_meta)
#     print(f"Lm head shape: {lm_head.weight.shape}")
#     print(f"Lm head Weight | Max: {lm_head.weight.max().item():.4f} "
#       f"| Min: {lm_head.weight.min().item():.4f} "
#       f"| Mean: {lm_head.weight.mean().item():.4f} "
#       f"| Std: {lm_head.weight.std().item():.4f}")


def print_weight_stats(name, tensor, warn_std=0.1):
    if tensor is None:
        print(f"{name} | NONE"); return
    if tensor.is_meta:
        print(f"{name} | META"); return

    data = tensor.detach().float()
    flags = "".join([
        " | NaN"      if torch.isnan(data).any()       else "",
        " | INF"      if torch.isinf(data).any()       else "",
        " | HIGH_STD" if data.std().item() > warn_std  else "",
    ])
    print(
        f"{name} | Shape: {tuple(data.shape)} "
        f"| Max: {data.max().item():.4f} | Min: {data.min().item():.4f} "
        f"| Mean: {data.mean().item():.4f} | Std: {data.std().item():.4f}{flags}"
    )


# ─────────────────────────────────────────────
# Generic checkers
# ─────────────────────────────────────────────

def check_attention(attn, prefix=""):
    """Supports Qwen/Qwen2, M2M100, BART, T5-like."""
    for attr in ("q_proj", "k_proj", "v_proj"):
        if hasattr(attn, attr):
            proj = getattr(attn, attr)
            print_weight_stats(f"{prefix}.{attr}.weight", proj.weight)
            if getattr(proj, "bias", None) is not None:
                print_weight_stats(f"{prefix}.{attr}.bias", proj.bias)

    # output projection (Qwen → o_proj, BART/M2M → out_proj)
    out_attr = "o_proj" if hasattr(attn, "o_proj") else "out_proj" if hasattr(attn, "out_proj") else None
    if out_attr:
        print_weight_stats(f"{prefix}.{out_attr}.weight", getattr(attn, out_attr).weight)


def check_mlp(mlp, prefix=""):
    for attr in ("gate_proj", "up_proj", "down_proj"):
        print_weight_stats(f"{prefix}.{attr}.weight", getattr(mlp, attr).weight)


def check_norm(norm, prefix=""):
    for attr in ("weight", "bias"):
        val = getattr(norm, attr, None)
        if val is not None:
            print_weight_stats(f"{prefix}.{attr}", val)


def check_fc_pair(layer, prefix=""):
    """Check fc1/fc2 (used by M2M100 encoder/decoder FFN)."""
    print_weight_stats(f"{prefix}.fc1.weight", layer.fc1.weight)
    print_weight_stats(f"{prefix}.fc2.weight", layer.fc2.weight)


# ─────────────────────────────────────────────
# Transformer-layer checkers
# ─────────────────────────────────────────────

def check_qwen_layer(layer, prefix=""):
    check_mlp(layer.mlp, f"{prefix}.mlp")
    check_norm(layer.input_layernorm,          f"{prefix}.input_layernorm")
    check_norm(layer.post_attention_layernorm, f"{prefix}.post_attention_layernorm")


def check_m2m_encoder_layer(layer, idx):
    p = f"mt_encoder.layers.{idx}"
    check_fc_pair(layer, p)
    check_norm(layer.self_attn_layer_norm, f"{p}.self_attn_layer_norm")
    check_norm(layer.final_layer_norm,     f"{p}.final_layer_norm")


def check_m2m_decoder_layer(layer, idx):
    p = f"mt_decoder.layers.{idx}"
    check_attention(layer.encoder_attn,    f"{p}.encoder_attn")
    check_fc_pair(layer, p)
    check_norm(layer.self_attn_layer_norm,   f"{p}.self_attn_layer_norm")
    check_norm(layer.encoder_attn_layer_norm,f"{p}.encoder_attn_layer_norm")
    check_norm(layer.final_layer_norm,       f"{p}.final_layer_norm")


# ─────────────────────────────────────────────
# Component checkers
# ─────────────────────────────────────────────

def _section(title, width=80):
    print(f"\n{'=' * width}\n{title}\n{'=' * width}")


def check_qwen_decoder(model):
    _section("CHECK QWEN LLM")
    qwen = model.llm.model
    print_weight_stats("llm.embed_tokens.weight", qwen.embed_tokens.weight)
    for idx, layer in enumerate(qwen.layers):
        print(f"\n{'#' * 10} QWEN LAYER {idx} {'#' * 10}")
        check_qwen_layer(layer, prefix=f"llm.layers.{idx}")
    check_norm(qwen.norm, prefix="llm.final_norm")
    print_weight_stats("llm.lm_head.weight", model.llm.lm_head.weight)


def check_connector(model):
    _section("CHECK CONNECTOR")
    conn = model.connector
    for i, attr in enumerate(("linear_1", "linear_2")):
        print_weight_stats(
            f"connector.encoder_project.{attr}.weight",
            getattr(conn.encoder_project, attr).weight,
        )
    print_weight_stats(
        "connector.post_encoder.embed_tokens.weight",
        conn.post_encoder.embed_tokens.weight,
    )
    for idx, layer in enumerate(conn.post_encoder.layers):
        print(f"\n{'#' * 10} CONNECTOR LAYER {idx} {'#' * 10}")
        check_qwen_layer(layer, prefix=f"connector.post_encoder.layers.{idx}")
    check_norm(conn.post_encoder.norm, prefix="connector.post_encoder.norm")


def check_m2m100(model):
    _section("CHECK M2M100")
    mt = model.mt_model.model
    print_weight_stats("mt_model.shared.weight", mt.shared.weight)

    _section("M2M100 ENCODER", width=50)
    for idx, layer in enumerate(mt.encoder.layers):
        print(f"\n{'#' * 10} ENCODER LAYER {idx} {'#' * 10}")
        check_m2m_encoder_layer(layer, idx)
    check_norm(mt.encoder.layer_norm, prefix="mt_encoder.layer_norm")

    _section("M2M100 DECODER", width=50)
    for idx, layer in enumerate(mt.decoder.layers):
        print(f"\n{'#' * 10} DECODER LAYER {idx} {'#' * 10}")
        check_m2m_decoder_layer(layer, idx)
    check_norm(mt.decoder.layer_norm, prefix="mt_decoder.layer_norm")

    print_weight_stats("mt_model.lm_head.weight", model.mt_model.lm_head.weight)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def check_weight(model):
    print(f"\n{'#' * 100}\nFULL MODEL WEIGHT CHECK\n{'#' * 100}")
    check_qwen_decoder(model)
    check_connector(model)
    check_m2m100(model)
    _section("CHECK FUSE MODEL")
    check_norm(model.fuse_model.layer_norm, prefix="fuse_model.layer_norm")
    print("\n✅ DONE CHECKING WEIGHTS")