# Receipt Analyzer 

This project demonstrates the OCR pipeline to extract the key information in Vietnamese supermarket invoices including name, address, order ID, purchased date and a list of purchased items. Besides, the pipeline also supports user to retrieve the desire item with its quantity and price. The project is composed of YOLOv8 - the state-of-the-art model for object detection - to localize the text region, VietOCR - a OCR model is trained on large Vietnamese text dataset - to recognize Vietnamese texts in detected regions, LayoutLMv3 - the modern and effective multimodal architecture in KIE task - to extract the key information.    

## Guide to use

Clone this repository on your local environment.
```sh
git clone https://github.com/ChrisPham-0502/Receipt_Analyzer.git
```

Then, install some necessary libraries:
```sh
!pip install -r requirements.txt
```

  Finally, run the following code to inference:
```sh
!python app.py
```
