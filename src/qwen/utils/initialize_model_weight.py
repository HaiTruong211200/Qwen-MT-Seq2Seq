import torch
import torch.nn as nn

from qwen.models.modules.normalization import QwenRMSNorm

def manual_fix_connector_weights(model, target_dim=1024):
    print("\n💉 ĐANG TIÊM GIÁ TRỊ KHỞI TẠO VÀO CONNECTOR (MANUAL FIX)...")
    
    # 1. Định vị Connector
    # Tuỳ vào cách bạn đặt tên, hãy trỏ đúng vào connector
    try:
        post_encoder = model.get_encoder().connector.post_encoder
    except AttributeError:
        print("❌ Không tìm thấy đường dẫn model.get_encoder().connector.post_encoder")
        return

    # 2. Duyệt qua từng module con trong post_encoder
    count_fixed = 0
    for name, module in post_encoder.named_modules():
        
        # --- FIX RMSNORM ---
        if isinstance(module, QwenRMSNorm):
            # Ép về 1.0 bất chấp
            with torch.no_grad():
                module.weight.fill_(1.0)
            count_fixed += 1
            
        # --- FIX LINEAR ---
        elif isinstance(module, nn.Linear):
            # Kaiming Init cho Weight
            with torch.no_grad():
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # Zero Init cho Bias (nếu có)
                if module.bias is not None:
                    module.bias.fill_(0.0)
            count_fixed += 1
            
        # --- FIX EMBEDDING (Nếu chưa xóa) ---
        elif isinstance(module, nn.Embedding):
            with torch.no_grad():
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    module.weight[module.padding_idx].fill_(0.0)
            count_fixed += 1
    decoder_layers = model.get_decoder().layers
    for name, module in decoder_layers.named_modules():
    
    # --- FIX RMSNORM ---
        if isinstance(module, QwenRMSNorm):
            # Ép về 1.0 bất chấp
            with torch.no_grad():
                module.weight.fill_(1.0)
            count_fixed += 1
            
        # --- FIX LINEAR ---
        elif isinstance(module, nn.Linear):
            # Kaiming Init cho Weight
            with torch.no_grad():
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # Zero Init cho Bias (nếu có)
                if module.bias is not None:
                    module.bias.fill_(0.0)
            count_fixed += 1
            
        # --- FIX EMBEDDING (Nếu chưa xóa) ---
        elif isinstance(module, nn.Embedding):
            with torch.no_grad():
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    module.weight[module.padding_idx].fill_(0.0)
            count_fixed += 1
    decoder_norm = model.get_decoder().norm
    if isinstance(decoder_norm, QwenRMSNorm):
        with torch.no_grad():
                decoder_norm.weight.fill_(1.0)
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head
        current_in_dim = lm_head.in_features
        vocab_size = lm_head.out_features
        
        print(f"🧐 Kiểm tra LM_HEAD: Hiện tại In={current_in_dim} | Out={vocab_size}")
        
        # Nếu kích thước bị sai (Ví dụ: đang là 2048 mà bạn muốn 1024)
        # if current_in_dim != target_dim:
        print(f"🚨 PHÁT HIỆN SAI KÍCH THƯỚC! Đang thay thế Linear({current_in_dim}) -> Linear({target_dim})...")
        
        # TẠO LỚP MỚI ĐÚNG KÍCH THƯỚC
        new_lm_head = nn.Linear(target_dim, vocab_size, bias=False)
        
        # Chuyển sang đúng device/dtype của model
        new_lm_head = new_lm_head.to(lm_head.weight.device).to(lm_head.weight.dtype)
        
        # Thay thế vào model
        model.lm_head = new_lm_head
        print("✅ Đã thay thế lm_head thành công.")
        
        # KHỞI TẠO TRỌNG SỐ LM_HEAD (Dù mới hay cũ cũng phải init lại)
        print("💉 Đang khởi tạo trọng số cho lm_head...")
        with torch.no_grad():
            nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
            
    else:
        print("❌ Model không có lớp lm_head.")
    

    print(f"✅ ĐÃ TIÊM THUỐC THÀNH CÔNG CHO {count_fixed} LỚP TRONG POST_ENCODER.")

    # --- KIỂM TRA LẠI NGAY LẬP TỨC ---
    print("\n🔎 KẾT QUẢ SOI KÍNH HIỂN VI:")
    
    # Check 1 lớp Linear bất kỳ trong post_encoder
    for name, module in post_encoder.named_modules():
        if isinstance(module, nn.Linear):
            max_val = module.weight.max().item()
            std_val = module.weight.std().item()
            print(f"  Layer: {name}")
            print(f"  -> Max: {max_val:.4f} (Kỳ vọng ~0.1 - 0.5)")
            print(f"  -> Std: {std_val:.4f} (Kỳ vọng ~0.02)")
            
            if max_val > 100:
                print("❌ VẪN CÒN RÁC! (LỖI CỰC KỲ LẠ)")
            else:
                print("✅ SẠCH SẼ!")
            break # Check 1 cái là đủ
