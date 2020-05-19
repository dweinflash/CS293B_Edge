#!/usr/bin/env python3

import socket
import time
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

HOST = '192.168.0.13'  # Pi IP Address
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)


def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


def run_classify(model):

  # EfficientNet - Rock, Paper, Scissors custom dataset
  if (model == "rps"):
    labels = load_labels("labels_EfficientNet_rps.txt")
    interpreter = Interpreter("model_EfficientNet_rps.tflite")
  # MobileNet - ImageNet
  else:
    labels = load_labels("labels_mobilenet_quant_v1_224.txt")
    interpreter = Interpreter("mobilenet_v1_1.0_224_quant.tflite")

  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  # Classify img.jpeg in current directory
  image = Image.open("img.jpeg").convert('RGB').resize((width, height), Image.ANTIALIAS)
  start_time = time.time()
  results = classify_image(interpreter, image)
  elapsed_ms = (time.time() - start_time) * 1000
  label_id, prob = results[0]
  annotate_text = '%s,%.2f,%.1f' % (labels[label_id], prob,
                                                    elapsed_ms)
  return annotate_text


def main():
    SOCK = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    SOCK.bind((HOST, PORT))
    SOCK.listen()
    conn, addr = SOCK.accept()
    
    # Handle 'Classify' and 'Exit' requests
    while True:
        data = conn.recv(5)
        if (str(data.decode()) == 'clas0'):
            res = run_classify("rps")
            print(res)
            conn.sendall(res.encode())
        elif (str(data.decode()) == 'clas1'):
            res = run_classify("ImageNet")
            print(res)
            conn.sendall(res.encode())
        elif (str(data.decode()) == "exit!"):
            print("Exit")
            break
    
    conn.close()
    SOCK.close()


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        conn.close()
        SOCK.close()