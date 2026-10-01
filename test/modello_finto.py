# Modello FINTO solo per provare la pipeline del browser: classe 1 = "verde meno rosso".
import numpy as np, onnx
from onnx import helper, TensorProto as T
x = helper.make_tensor_value_info('input', T.FLOAT, [1, 3, 192, 320])
y = helper.make_tensor_value_info('logits', T.FLOAT, [1, 2, 192, 320])
nodes = [
  helper.make_node('Split', ['input'], ['r', 'g', 'b'], axis=1, num_outputs=3),
  helper.make_node('Sub', ['g', 'r'], ['d0']),
  helper.make_node('Sub', ['d0', 'off'], ['d']),
  helper.make_node('Sub', ['d', 'd'], ['z']),
  helper.make_node('Concat', ['z', 'd'], ['logits'], axis=1),
]
off = helper.make_tensor('off', T.FLOAT, [], [0.5])   # le bande nere della lettera di casella restano sfondo
g = helper.make_graph(nodes, 'finto', [x], [y], initializer=[off])
m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 18)]); m.ir_version = 9
onnx.checker.check_model(m); onnx.save(m, 'guida_finto.onnx'); print('ok')
