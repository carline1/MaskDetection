from peewee import *
import cv2
import base64
import numpy as np

db = PostgresqlDatabase(
    database="mask_detect_db",
    user="postgres",
    password="root"
)


class BaseModel(Model):
    class Meta:
        database = db


class User(BaseModel):
    fio = CharField(max_length=80)
    img = BlobField()


def add_new_user(fio: str, img):
    img = encode_img_to_bytes(img)
    user = User(fio=fio, img=img)
    user.save()
    user_id = user.id
    return user_id


def encode_img_to_bytes(img):
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text


def decode_bytes_to_img(bytes):
    jpg_original = base64.b64decode(bytes)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    return img


User.create_table()
