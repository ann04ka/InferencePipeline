Время рассчета одного изображения 200х200 на cpu, время усреднено по 20 примерам ( Intel Celeron) :

make_inference inference:  0.053 c

OpenCV Python API inference:  0.126 c

OnnxRuntime Python API inference:  0.030 c

OpenVINO inference:  0.032 c



* [ ] Прописать девайс  CPU
* [ ] Описать архитектуру модели
* [ ] Загрузить в модель актуальные веса
* [ ] Загрузить пример входных данных / Создать torch.randn(input.shape())


* [ ] Экспортировать модель в onnx (подробнее <https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html>)

  `/пример`

  `def exportONNX(model):   `

` torch_input = torch.randn(1, 1, 200, 200)`

` model_path = 'sourse_model_name.onnx'`

`torch.onnx.export(model, torch_input, model_path,`

`verbose=True,    input_names=["input"],    output_names=["output"],`

`    opset_version=11)`

После экспорта модели  в onnx можно проверить правильность экпорта не только по выводу в консоли, но и отрисовав граф модели с помощью netron <https://netron.app/>, ниже прикрепляю пример



На данный момент есть ~~2~~ 3  пути дальнейшего запуска модели

* Путь OpenCV Python API (подробнее <https://docs.opencv.org/4.x/dc/d70/pytorch_cls_tutorial_dnn_conversion.html>) (c++ <https://docs.opencv.org/4.x/dd/d55/pytorch_cls_c_tutorial_dnn_conversion.html>)
  * [ ] Загрузить модель из Onnx
    `opencv_net = cv2.dnn.readNetFromONNX(onnx_model_path)`
  * [ ] Экспортировать входные данные в dnn.blob

    `input_blob = cv2.dnn.blobFromImage(image=input_img, size=(200, 200))`
  * [ ] Загрузить в экземпляр модели экспортированные данные

    `opencv_net.setInput(input_blob)`
  * [ ] Запустить модель

    `out = opencv_net.forward()`
* Путь OnnxRuntime (подробнее <https://onnxruntime.ai/docs/api/python/api_summary.html>) (c++ <https://onnxruntime.ai/docs/get-started/with-cpp.html#samples>)
  * [ ] Загрузить модель из Onnx

  `session = onnxruntime.InferenceSession('best_model_landmarks.onnx', None, providers=['CPUExecutionProvider'])`
  * [ ] Определить входные и выходные данные

  `input_nodes = session.get_inputs()`

  `input_names = [node.name for node in input_nodes]`

  `input_shapes = [node.shape for node in input_nodes]`

  `input_types = [node.type for node in input_nodes]`

  `output_nodes = session.get_outputs()`

  `output_names = [node.name for node in output_nodes]`

  `output_shapes = [node.shape for node in output_nodes]`

  `output_types = [node.type for node in output_nodes]`
  * [ ] Запустить модель

  `output_tensors = session.run([], input_feed={input_names[0]: input_img}, run_options=None)`
* Путь OpenVINO (подробнее <https://docs.openvino.ai/2022.3/notebooks/002-openvino-api-with-output.html#doing-inference-on-a-model>) (c++ <https://docs.openvino.ai/2022.3/openvino_docs_get_started_get_started_demos.html>)
  * [ ] Создать ядро  OpenVINO

  `    ie = Core()`
  * [ ] Загрузить модель из Onnx
    `model_onnx = ie.read_model(model=onnx_model_path)`
    `compiled_model = ie.compile_model(model=model_onnx, device_name="CPU")`
  * [ ] Определить входные и выходные данные
    `input_layer = compiled_model.input(0)`
    `output_layer = compiled_model.output(0)`
  * [ ] Запустить модель
    `request = compiled_model.create_infer_request()`
    `request.infer(inputs={input_layer.any_name: input_data})`
    `result = request.get_output_tensor(output_layer.index).data`
