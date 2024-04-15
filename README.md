# Airbus

The YOLOv8 model is trained to search for aircraft on the ground.
This was done at [Colab](https://colab.research.google.com/drive/1RxXaa6zn_ZOQJMDdQOjoELw6qh3ae3pH?usp=sharing)

After that, it was exported to tflite format and launched on google coral. It works.

https://coral.ai/docs/accelerator/get-started

for all files path

python3 ./detect_airplanes_coral.py   --model ./best_full_integer_quant_edgetpu.tflite   --input ~/googledrive/neuro/airbus/extras

or for one file

python3 ./detect_airplanes_coral.py   --model ./best_full_integer_quant_edgetpu.tflite   --input ./495b73c8-024f-46cc-b426-05e49bbe5074.jpg

or without coral TPU

python3 ./detect_airplanes_tflite.py  --model ./best_float32.tflite    --input ./495b73c8-024f-46cc-b426-05e49bbe5074.jpg
Coral
![Coral](https://github.com/palich2000/Airbus/blob/main/495b73c8-024f-46cc-b426-05e49bbe5074_result.png)
TfLite
![TfLite](https://github.com/palich2000/Airbus/blob/main/495b73c8-024f-46cc-b426-05e49bbe5074_result_tflite.png)
