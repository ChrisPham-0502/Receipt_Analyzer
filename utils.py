import numpy as np
from datasets import load_metric
from PIL import ImageDraw, ImageFont
import pandas as pd


metric = load_metric("seqeval")


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000)
    ]


def normalize_box(bbox, width, height):
    return [
        int((bbox[0] / width) * 1000),
        int((bbox[1] / height) * 1000),
        int((bbox[2] / width) * 1000),
        int((bbox[3] / height) * 1000)
    ]


def draw_output(image, true_predictions, true_boxes):
    def iob_to_label(label):
        label = label
        if not label:
            return 'other'
        return label

    # width, height = image.size

    # predictions = logits.argmax(-1).squeeze().tolist()
    # is_subword = np.array(offset_mapping)[:,0] != 0
    # true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    # true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]

    # draw
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline='red')
        draw.text((box[0] + 10, box[1] - 10),
                  text=predicted_label, fill='red', font=font)

    return image


def create_df(true_texts,
              true_predictions,
              chosen_labels=['SHOP_NAME', 'ADDR', 'TITLE', 'PHONE',
                             'PRODUCT_NAME', 'AMOUNT', 'UNIT', 'UPRICE', 'SUB_TPRICE', 'UDISCOUNT',
                             'TAMOUNT', 'TPRICE', 'FPRICE', 'TDISCOUNT',
                             'RECEMONEY', 'REMAMONEY',
                             'BILLID', 'DATETIME', 'CASHIER']
              ):

    data = {'text': [], 'class_label': [], 'product_id': []}
    product_id = -1
    for text, prediction in zip(true_texts, true_predictions):
        if prediction not in chosen_labels:
            continue

        if prediction == 'PRODUCT_NAME':
            product_id += 1
            

        if prediction in ['AMOUNT', 'UNIT', 'UDISCOUNT', 'UPRICE', 'SUB_TPRICE',
                          'UDISCOUNT', 'TAMOUNT', 'TPRICE', 'FPRICE', 'TDISCOUNT',
                          'RECEMONEY', 'REMAMONEY']:
            text = reformat(text)


        if prediction in ['AMOUNT', 'SUB_TPRICE', 'UPRICE', 'PRODUCT_NAME']:
            data['product_id'].append(product_id)
        else:
            data['product_id'].append('')


        data['class_label'].append(prediction)
        data['text'].append(text)


    df = pd.DataFrame(data)

    return df


def reformat(text: str):
    try:
        text = text.replace('.', '').replace(',', '').replace(':', '').replace('/', '').replace('|', '').replace(
            '\\', '').replace(')', '').replace('(', '').replace('-', '').replace(';', '').replace('_', '')
        return int(text)
    except:
        return text

def find_product(product_name, df):
    product_name = product_name.lower()
    product_df = df[df['class_label'] == 'PRODUCT_NAME']
    mask = product_df['text'].str.lower().str.contains(product_name, case=False, na=False)
    if mask.any():
        product_id = product_df.loc[mask, 'product_id'].iloc[0]
        product_info = df[df['product_id'] == product_id]
        
        prod_name = product_info.loc[product_info['class_label'] == 'PRODUCT_NAME', 'text'].iloc[0]
            
        try:
            amount = product_info.loc[product_info['class_label'] == 'AMOUNT', 'text'].iloc[0]
        except:
            print("Error: cannot find amount")
            amount = ''
            
        try:
            uprice = product_info.loc[product_info['class_label'] == 'UPRICE', 'text'].iloc[0]
        except:
            print("Error: cannot find unit price")
            uprice = ''
            
        try:
            sub_tprice = product_info.loc[product_info['class_label'] == 'SUB_TPRICE', 'text'].iloc[0]
        except:
            print("Error: cannot find sub total price")
            sub_tprice = ''
            
        #print("Sản phẩm: ", product_info.loc[product_info['class_label'] == 'PRODUCT_NAME', 'text'].iloc[0])
        #print("Số lượng: ", product_info.loc[product_info['class_label'] == 'AMOUNT', 'text'].iloc[0])
        #print("Đơn giá: ", product_info.loc[product_info['class_label'] == 'UPRICE', 'text'].iloc[0])
        #print("Thành tiền: ", product_info.loc[product_info['class_label'] == 'SUB_TPRICE', 'text'].iloc[0])
        return f"Sản phẩm: {prod_name}\n Số lượng: {amount}\n Đơn giá: {uprice}\n Thành tiền: {sub_tprice}"
    else:
        #print("Không tìm thấy item nào phù hợp.")
        return "Không tìm thấy item nào phù hợp."
    #return result = product_df['text'].str.contains(product_name, case=False, na=False).any()
    #return product_df[product_df['text'].str.contains(product_name, case=False, na=False)]


def get_info(df):
    try:
        shop_name = df.loc[df['class_label'] == 'SHOP_NAME', 'text'].iloc[0]
    except:
        print("Error: cannot find shop name")
        shop_name = ''
    print("Tên siêu thị: ", shop_name)

    try:
        addr = df.loc[df['class_label'] == 'ADDR', 'text'].iloc[0]
    except:
        print("Error: cannot find address")
        addr = ''
    print("Địa chỉ: ", addr)

    try:
        bill_id = df.loc[df['class_label'] == 'BILLID', 'text'].iloc[0]
    except:
        print("Error: cannot find bill id")
        bill_id = ''
    print("ID hóa đơn: ", bill_id)

    try:
        date_time = df.loc[df['class_label'] == 'DATETIME', 'text'].iloc[0]
    except:
        print("Error: cannot find date and time")
        date_time = ''
    print("Ngày: ", date_time)

    try:
        cashier = df.loc[df['class_label'] == 'CASHIER', 'text'].iloc[0]
    except:
        print("Error: cannot find cashier")
        cashier = ''
    print("Nhân viên: ", cashier)

    return f"Tên siêu thị: {shop_name}\n Địa chỉ: {addr}\n ID hóa đơn: {bill_id}\n Ngày: {date_time}\n Nhân viên: {cashier}\n"
