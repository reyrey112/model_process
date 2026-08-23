import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    "C:/Users/reyde/Desktop/Coding_Project/Portfolio/model_process/models/VAE_industrial", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Check input/output names and shapes (useful for sanity-checking)
for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)
for out in session.get_outputs():
    print(out.name, out.shape, out.type)

outputs = [x.name for x in session.get_outputs()]
outputs = [x.name for x in session.get_inputs()]

print(outputs)

input_name = session.get_inputs()[0].name

single_row = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]], dtype=np.float32)

single_row = single_row.reshape(1,1,24)

outputs = session.run(
    ["output_name", "output_mu", "output_logvar"],
    {"input_name": single_row}
)
output, mu, logvar = outputs
print("output:", output.shape)
print("mu:", mu.shape)
print("logvar:", logvar.shape)

print(output)