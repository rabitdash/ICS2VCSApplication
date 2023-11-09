import streamlit as st
import numpy as np
import torch
import logging
import cv2
from os.path import splitext
logger = logging.getLogger("MRVCS Decoder")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
import torch
import scipy.io as scio
import argparse
import numpy as np
# from thop import profile
import imageio
import os
# %% Encode
from CSNet import CSNet as HmrCSNet
from CSNet import save_name as hr_save_name
use_cuda = True
device = torch.device("cuda:0" if (
    use_cuda and torch.cuda.is_available()) else "cpu")
hrmodel = HmrCSNet().to(device)
# hrmodel.load_state_dict(torch.load(
#     "./" + hr_save_name, map_location=torch.device(device)))
hrmodel.load_state_dict(torch.load(
    hr_save_name, map_location=torch.device(device)))
hrmodel.eval()
model = hrmodel
VIDEO_WIDTH = 704
VIDEO_HEIGHT = 576
CACHE_SIZE=8

def fimage2uimage(lr_rec):
    lr_rec[lr_rec > 1] = 1
    lr_rec[lr_rec < 0] = 0
    lr_rec = (lr_rec * 255).astype(np.uint8)
    return  lr_rec

def encode(pic: np.ndarray):
    # RG
    # GB
    global bayer
    with torch.no_grad():
        # buffer =
        pic_t = torch.from_numpy(pic/255.0).unsqueeze(0).permute([0,3,1,2]).float().to(device)
        meas_t = model.compress(pic_t)
        # else:
        #     # 转换为RGB
        #     pic_t[:,:,:,:] = pic_t[:,:,:,:]

        return meas_t



def decode(meas: np.ndarray):
    # RG
    # GB
    with torch.no_grad():
        # out = out_pic1[0, :, :, :, :].cpu().numpy()
        # imageio.imwrite("ff0.jpg", cv2.cvtColor(np.transpose(out[0], (1, 2, 0)), cv2.COLOR_BGR2RGB))
        # print(out.shape)
        meas = torch.from_numpy(meas).float().to(device)
        out=model.recon(meas)
        return out
def ICS_Compress(source_file_path, output_file_path=None):
    vpic = st.empty()
    text = st.empty()
    bar = st.progress(0)
    # 捕获视频
    stem=f"ICS"
    if output_file_path is None:
        output_file_path = "./tmp/" + stem+".mat"
    print(f"Write to {output_file_path}")
    cap = cv2.VideoCapture(source_file_path)
    # 定义编解码器，创建VideoWriter 对象
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # FPS = 2
    # out = cv2.VideoWriter(output_file_path, fourcc, FPS, (450, 444), False)
    # （写出的文件，？？，帧率，（分辨率），是否彩色）  非彩色要把每一帧图像装换成灰度图

    # 仅对视频有用，统计视频总帧数
    SOURCE_FRAME_NUM = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    PROCESSED_FRAME_COUNT = 0
    if os.path.exists(output_file_path):
        with open(output_file_path,'w'):
            pass            # 清空文件

    mat_file = open(output_file_path,"ab") # 以添加方式写入
    while (cap.isOpened()):
        ret, frame = cap.read()
        # logging.info(f"P {PROCESSED_FRAME_COUNT}/{SOURCE_FRAME_NUM}")
        if ret == True:
            # frame = cv2.flip(frame,0)  #可以进行视频反转
            # write the flipped frame
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #换换成灰度图
            frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
            I = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vpic.image(I)
            meas = encode(I)
            
            # out.write(quantize(meas))
            PROCESSED_FRAME_COUNT += 1
            text.text(f"编码帧 {PROCESSED_FRAME_COUNT}/{SOURCE_FRAME_NUM}")
            bar.progress(int(PROCESSED_FRAME_COUNT/SOURCE_FRAME_NUM*100))
            scio.savemat(mat_file,{f"frame_{PROCESSED_FRAME_COUNT}":meas.cpu().numpy()},do_compression=True)
        else:
            cap.release()
    mat_file.close()

    # Release everything if job is finished
    print("release")
    return output_file_path

def ICS_Recon(source_file_path, output_file_path=None, ):
    snap = st.empty()
    vpic = st.empty()
    text = st.empty()
    bar = st.progress(0)

    if output_file_path is None:
        stem = "Recon"
        output_file_path = "./tmp/" + stem + f"_MATRE.mp4"
    # cap = cv2.VideoCapture(source_file_path)
    # # 定义编解码器，创建VideoWriter 对象
    meas_mat = scio.loadmat(source_file_path)
    meas_mat_keys = meas_mat.keys()
    # # （写出的文件，？？，帧率，（分辨率），是否彩色）  非彩色要把每一帧图像装换成灰度图

    # 仅对视频有用，统计视频总帧数

    # VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    SOURCE_FRAME_NUM = len(meas_mat_keys)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    FPS = 24
    out = cv2.VideoWriter(output_file_path, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT), True)

    PROCESSED_FRAME_COUNT = 0
    for i in meas_mat_keys:
        # print(frame.shape)
        if 'frame' not in i:
            continue
        meas = meas_mat[i]
        PROCESSED_FRAME_COUNT += 1
        logger.info(f"DECODING {PROCESSED_FRAME_COUNT}/{SOURCE_FRAME_NUM}")
        text.text(f"解码帧 {PROCESSED_FRAME_COUNT}/{SOURCE_FRAME_NUM}")
        bar.progress(int(PROCESSED_FRAME_COUNT/SOURCE_FRAME_NUM*100))

        # frame = (frame * 255).astype(np.uint8)
        # frame = frame / 255.0
        recon_frame = decode(meas)
        # cv2.imshow("recon",meas_re[0])
        # key = cv2.waitKey(20)
        # if key == ord('q'):
        #     break
        logger.debug(f"Writing recon")
        write_frame = fimage2uimage(recon_frame[0].permute([1,2,0]).cpu().numpy())
        vpic.image(write_frame)
        write_frame = cv2.cvtColor(write_frame, cv2.COLOR_RGB2BGR)
        # print(meas_re.shape)
        out.write(write_frame)

    logger.warning("Video release.")
    out.release()
    return  output_file_path

# %% App GUI definition

# if __name__ == '__main__':
#     SCI_Compress("bosphorus_dataset.mp4")
#     SCI_Recon("bosphorus_dataset_CS_B8.mat")

mcol1, mcol2 = st.columns(2)
with mcol1:
    st.text("视频编码")
    raw_video = st.file_uploader("原始视频", type='.mp4')
    print(type(raw_video))

    enc_btn = st.button("压缩")
    if enc_btn:
        raw_video = raw_video.getvalue()
        with open('./raw.mp4', 'wb') as f:
            f.write(raw_video)
        CSVideo_path = ICS_Compress("./raw.mp4")
        st.info("压缩完成")
        # vf = open(CSVideo_path, 'rb').read()
        # matf =open(CSMat_path, 'rb').read()
        with open(CSVideo_path, 'rb') as vf:
            st.download_button(label="压缩结果",
                               data=vf.read(),
                               file_name=CSVideo_path)

with mcol2:
    st.text("视频重构")
    mat = st.file_uploader("压缩快照数据.mat", type='.mat')
    recon_btn = st.button("重构")

    if recon_btn:
        mat=mat.getvalue()
        with open('./tmp/cs.mat', 'wb') as f:
            f.write(mat)
        RecVideo_path = ICS_Recon('./tmp/cs.mat')
        st.info("重构完成")
        # vf = open(CSVideo_path, 'rb').read()
        # matf =open(CSMat_path, 'rb').read()
        with open(RecVideo_path, 'rb') as vf:
            st.download_button(label="重构结果",
                               data=vf.read(),
                               file_name=RecVideo_path)



