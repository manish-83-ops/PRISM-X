"""Export cross-encoder/ms-marco-MiniLM-L-6-v2 to ONNX FP32 and Dynamic INT8."""

import os
import sys
import time
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models" / "onnx_cross_encoder"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ONNX_FP32_PATH = MODELS_DIR / "model_fp32.onnx"
ONNX_INT8_PATH = MODELS_DIR / "model_int8.onnx"

def main():
    print(f"Loading {MODEL_NAME} for ONNX export...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    dummy_query = "where did hip hop rap come from"
    dummy_doc = "Hip hop originated in the South Bronx in the 1970s."
    inputs = tokenizer(
        [[dummy_query, dummy_doc]],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    }

    print(f"Exporting FP32 ONNX model to {ONNX_FP32_PATH}...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
            str(ONNX_FP32_PATH),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"FP32 export completed. Size: {ONNX_FP32_PATH.stat().st_size / (1024*1024):.2f} MB")

    # Quantize to dynamic INT8 using ONNX Runtime quantization
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        print(f"Quantizing to dynamic INT8 at {ONNX_INT8_PATH}...")
        quantize_dynamic(
            model_input=str(ONNX_FP32_PATH),
            model_output=str(ONNX_INT8_PATH),
            weight_type=QuantType.QInt8,
        )
        print(f"INT8 quantization completed. Size: {ONNX_INT8_PATH.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"ONNX Quantization error: {e}")

    # Validate ONNX model with ONNX Runtime
    import onnxruntime as ort
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 6
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session_fp32 = ort.InferenceSession(str(ONNX_FP32_PATH), sess_opts, providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
        "token_type_ids": inputs["token_type_ids"].numpy(),
    }

    ort_out_fp32 = session_fp32.run(None, ort_inputs)[0]

    with torch.no_grad():
        torch_out = model(**inputs).logits.numpy()

    diff_fp32 = abs(float(torch_out[0][0]) - float(ort_out_fp32[0][0]))
    print(f"PyTorch score: {torch_out[0][0]:.6f} | ONNX FP32 score: {ort_out_fp32[0][0]:.6f} | Diff: {diff_fp32:.8f}")
    assert diff_fp32 < 1e-4, f"FP32 numerical disparity too large: {diff_fp32}"
    print("FP32 ONNX verification PASS (diff < 1e-4)")

    if ONNX_INT8_PATH.exists():
        session_int8 = ort.InferenceSession(str(ONNX_INT8_PATH), sess_opts, providers=["CPUExecutionProvider"])
        ort_out_int8 = session_int8.run(None, ort_inputs)[0]
        diff_int8 = abs(float(torch_out[0][0]) - float(ort_out_int8[0][0]))
        print(f"PyTorch score: {torch_out[0][0]:.6f} | ONNX INT8 score: {ort_out_int8[0][0]:.6f} | Diff: {diff_int8:.6f}")
        print("INT8 ONNX model loaded successfully.")

if __name__ == "__main__":
    main()
