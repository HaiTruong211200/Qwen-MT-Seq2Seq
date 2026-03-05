def check_weight(model):
    print("--- KIỂM TRA TRỌNG SỐ Encoder RMS NORM ---")
    # Thay đường dẫn tới module project_down của bạn cho đúng
    # Ví dụ: model.encoder.connector.encoder_project.linear_1
    encoder = model.get_encoder()
    for idx, layer in enumerate(encoder.layers):
        print(f"Layer {idx}")
        print(
            f"Input RMS Weight | Max: {layer.input_layernorm.weight.max().item():.4f} "
            f"| Min: {layer.input_layernorm.weight.min().item():.4f} "
            f"| Mean: {layer.input_layernorm.weight.mean().item():.4f} "
            f"| Std: {layer.input_layernorm.weight.std().item():.4f}"
         )
        print(
            f"Post RMS Weight | Max: {layer.post_attention_layernorm.weight.max().item():.4f} "
            f"| Min: {layer.post_attention_layernorm.weight.min().item():.4f} "
            f"| Mean: {layer.post_attention_layernorm.weight.mean().item():.4f} "
            f"| Std: {layer.post_attention_layernorm.weight.std().item():.4f}"
         )
    rms_norm = model.get_encoder().norm 
    print(f"Encoder RMS norm Weight | Max: {rms_norm.weight.max().item():.4f} "
          f"| Min: {rms_norm.weight.min().item():.4f} "
          f"| Mean: {rms_norm.weight.mean().item():.4f} "
          f"| Std: {rms_norm.weight.std().item():.4f}")
    
    if rms_norm.weight.std().item() > 0.1:
        print("❌ CẢNH BÁO: Trọng số quá lớn! Hàm khởi tạo chưa chạy đúng.")
    else:
        print("✅ Trọng số có vẻ ổn (std nhỏ).")

    print("--- KIỂM TRA TRỌNG SỐ PROJECT DOWN ---")
    # Thay đường dẫn tới module project_down của bạn cho đúng
    # Ví dụ: model.encoder.connector.encoder_project.linear_1
    linear1 = model.get_encoder().connector.encoder_project.linear_1 
    print(f"Linear 1 Weight | Max: {linear1.weight.max().item():.4f} "
        f"| Min: {linear1.weight.min().item():.4f} "
        f"| Mean: {linear1.weight.mean().item():.4f} "
        f"| Std: {linear1.weight.std().item():.4f}")

    linear2 = model.get_encoder().connector.encoder_project.linear_2
    print(f"Linear 2 Weight | Max: {linear2.weight.max().item():.4f} "
        f"| Min: {linear2.weight.min().item():.4f} "
        f"| Mean: {linear2.weight.mean().item():.4f} "
        f"| Std: {linear2.weight.std().item():.4f}")
    
    if linear1.weight.std().item() > 0.1:
        print("❌ CẢNH BÁO: Trọng số quá lớn! Hàm khởi tạo chưa chạy đúng.")
    else:
        print("✅ Trọng số có vẻ ổn (std nhỏ).")

    print("--- KIỂM TRA TRỌNG SỐ CONNECTOR ---")
    for idx, layer in enumerate(encoder.connector.post_encoder.layers):
        print(f"Layer {idx}")
        print(
            f"Shape: {layer.input_layernorm.weight.shape}"
            f"Input RMS Weight | Max: {layer.input_layernorm.weight.max().item():.4f} "
            f"| Min: {layer.input_layernorm.weight.min().item():.4f} "
            f"| Mean: {layer.input_layernorm.weight.mean().item():.4f} "
            f"| Std: {layer.input_layernorm.weight.std().item():.4f}"
         )
        print(
            f"Shape: {layer.post_attention_layernorm.weight.shape}"
            f"Post RMS Weight | Max: {layer.post_attention_layernorm.weight.max().item():.4f} "
            f"| Min: {layer.post_attention_layernorm.weight.min().item():.4f} "
            f"| Mean: {layer.post_attention_layernorm.weight.mean().item():.4f} "
            f"| Std: {layer.post_attention_layernorm.weight.std().item():.4f}"
         )

    print("--- KIỂM TRA TRỌNG SỐ DECODER ---")
    decoder = model.get_decoder()
    for idx, layer in enumerate(decoder.layers):
        print(f"Layer {idx}")
        print(
            f"Shape: {layer.input_layernorm.weight.shape}"
            f"Input RMS Weight | Max: {layer.input_layernorm.weight.max().item():.4f} "
            f"| Min: {layer.input_layernorm.weight.min().item():.4f} "
            f"| Mean: {layer.input_layernorm.weight.mean().item():.4f} "
            f"| Std: {layer.input_layernorm.weight.std().item():.4f}"
         )
        print(
            f"Shape: {layer.post_attention_layernorm.weight.shape}"
            f"Post RMS Weight | Max: {layer.post_attention_layernorm.weight.max().item():.4f} "
            f"| Min: {layer.post_attention_layernorm.weight.min().item():.4f} "
            f"| Mean: {layer.post_attention_layernorm.weight.mean().item():.4f} "
            f"| Std: {layer.post_attention_layernorm.weight.std().item():.4f}")
    decoder_norm = decoder.norm
    print(f"Decoder RMS norm Weight | Max: {decoder_norm.weight.max().item():.4f} "
          f"| Min: {decoder_norm.weight.min().item():.4f} "
          f"| Mean: {decoder_norm.weight.mean().item():.4f} "
          f"| Std: {decoder_norm.weight.std().item():.4f}")
    
    lm_head = model.lm_head
    print(f"Lm head shape: {lm_head.weight.shape}")
    print(f"Lm head Weight | Max: {lm_head.weight.max().item():.4f} "
      f"| Min: {lm_head.weight.min().item():.4f} "
      f"| Mean: {lm_head.weight.mean().item():.4f} "
      f"| Std: {lm_head.weight.std().item():.4f}")