import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from hyejeong.demo.initializer2 import get_initializer
import base64
import io
import re

def set_image_ratio(im):
    ww, hh = im.size
    if (ww > hh):
        w = 512
        h = int(hh * (w/ww))
    if hh > ww:
        h = 512
        w = int(ww * (h/hh))
    return (w, h)

def image_to_base64(im):
    buffered = io.BytesIO()
    im.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ┌─────────────
# │ 초기화
# └─────────────
# Config set
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
# Style
st.markdown("""
<style>
    h1{
        text-align: center;
        color: grey;
    }
    div[data-testid="stHorizontalBlock"] > button{
        margin: 0px 30px;
    }
    div[data-testid="stElementContainer"]{
        text-align-last: center;
    }
    div.stButton > button {
        margin: -1rem 0px 0px 0px;
        padding: 1rem 5rem 1rem 4rem;
    }
    .stElementContainer iframe {
        position: relative;
        top: 0;
        left: 15%;
    }
    .stSlider {
        padding: 0px 2rem;
    }
    .st-emotion-cache-phe2gf li {
        margin: 0px;
    }
    ul {
        list-style-type: none;
        display: flex;
    }
    .card{
        margin-bottom: 10px
    }
    .box{
        padding: 10px;
        width: 262px;
        border: 2px solid lightgray;
        border-radius: 5px;
    }
    .index{
        display: flex;
        font-weight: bold;
    }
    .centered{
        text-align: center;
        margin-top: 5px;
    }
    img {
        width: 240px;
    }
</style>
            """, unsafe_allow_html=True)
# Class to Color v.v.
class_to_canvas = {
    "Bed":     "#FF0000",
    "Chair":   "#FF7F00",
    "Dresser": "#FFFF00",
    "Lamp":    "#00FF00",
    "Sofa":    "#0000FF",
    "Table":   "#9400D3"
}
canvas_to_class = {
    "#FF0000": "Bed",
    "#FF7F00": "Chair",
    "#FFFF00": "Dresser",
    "#00FF00": "Lamp",
    "#0000FF": "Sofa",
    "#9400D3": "Table"
}
class_dict = {
    "beds": "beds_db.sqlite3", 
    "chairs": "chairs_db.sqlite3", 
    "dressers": "dressers_db.sqlite3", 
    "sofas": "sofas_db.sqlite3", 
    "tables": "tables_db.sqlite3", 
    "lamps": "lamps_db.sqlite3"
}

# Title
st.markdown("<h1>🛋️ 모두의 인테리어 🛋️</h1></br></br>", unsafe_allow_html=True)
# Layout
left_padding, main, right_padding = st.columns([1, 6, 1], vertical_alignment='center')
left_main, right_main = main.columns(2, vertical_alignment='center')
# Set model
model = get_initializer()


# ┌─────────────
# │ 0. 조건 선택 구간
# └─────────────
with right_main:
    target       = st.file_uploader('Choose Image to upload…', type = (["jpg", "jpeg", "png"]))
    drawing_mode = st.radio("Drawing tool:", ("rect", "transform"), horizontal=True)
    class_name   = st.selectbox("Furniture:", ("Bed", "Chair", "Dresser", "Lamp", "Sofa", "Table"))
    prompt_style = st.selectbox("Style:", ("Contemporary", "Industry", "Mid-century", "Modern", "Rustic", "Scandinavian"))


# ┌─────────────
# │ 1. 사진이 올라와 있으면 -> BBOX + CLS + Text 정보 + Generation Option 입력 받기
# └─────────────
if target is None:
    model.reset()
elif target is not None:
    model.set_bg_image(target)

    with left_main:
        w, h = model.get_bg_image().size
        resize=350
        st.caption("Original Image")
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3,
            stroke_color=class_to_canvas[class_name],
            background_color="#EEEEEE",
            background_image=model.get_bg_image(resize=resize),
            update_streamlit=True,
            width=resize,
            height=resize,
            drawing_mode=drawing_mode,
            key="canvas",
        )
    with main:
        step = st.slider(
            "Inference step: ",
            min_value=25, max_value=100, value=50
        )
        btn_1 = st.button(
            "**GENERATION RUN**",
            icon=":material/assistant:",
            use_container_width=False
            )
        st.write('---')


# ┌─────────────
# │ 2. button을 누르면 -> 이미지 생성, 해당 이미지에 대한 유사도 기반 추천
# └─────────────       
    if btn_1:
        if canvas_result.json_data["objects"]:
            objects = pd.json_normalize(canvas_result.json_data["objects"])
            for col in objects.select_dtypes(include=['object']).columns:
                objects[col] = objects[col].astype("str")
            model.set_bbox_cls_info(objects)
            model.generation_run(prompt_style, step)

    pl, col_border, pr = main.columns([1,1,1])
    if model.get_ig_image() is not None:
        with col_border:
            st.image(
                model.get_ig_image(),
                caption="Generated Image",
                width=350
            )
            st.divider()
        
        for idx, cropped in enumerate(model.get_ig_crop_image()):
            lp, cc, rp = st.columns([1,8,1])
            col_4c = cc.columns([2,2,2,2])
            with st.container():
                top_3_items = model.search_run(idx)
                w, h = set_image_ratio(cropped)

                resized_cropped = cropped.resize((w, h), Image.LANCZOS)
                padded = ImageOps.pad(resized_cropped, (262, 262), color='#FFF')
                padded_base64 = image_to_base64(padded)


                col_4c[0].markdown(
                    f"""
                    <div class="box">
                        <ul>
                            <li>
                                <img src="data:image/png;base64,{padded_base64}"/>
                            </li>
                            <li>
                                <p class='centered'>Generated Image {idx + 1}</p>
                            </li>
                        </ul>
                    </div>

                    <div class="box card">
                        <ul>
                            <li>
                                <ul class="index">
                                    Code: 
                                </ul>
                            </li>
                            <li>
                                <ul class="index">
                                    Product name: 
                                </ul>
                            </li>
                            <li>
                                <ul class="index">
                                    Price:
                                </ul>
                            </li>
                            <li>
                                <ul class="index">
                                    Similarity: 
                                </ul>
                            </li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                for rank, (col, item) in enumerate(zip(col_4c[1:], top_3_items)):
                    name1 = re.split(r'(\d+)', item[1][2])
                    code = name1[1]
                    name2 = name1[2].split('-')
                    product_name = ' '.join(name2[1:])
                    if product_name is None:
                        product_name = ' '
                    image_path = f"/home/work/ModuInterior/dataset/bonn-custom/houzz/{item[1][-2]}/{item[1][1]}/{item[1][0]}"
                    similar_image = Image.open(image_path).convert('RGB')
                    similar_image_base64 = image_to_base64(similar_image)

                    col.markdown(
                        f"""
                        <div class="box">
                            <ul>
                                <li>
                                    <img src="data:image/png;base64,{similar_image_base64}"/>
                                </li>
                                <li>
                                    <p class='centered'>Rank {rank + 1}</p>
                                </li>
                            </ul>
                        </div>

                        <div class="box card">
                            <ul>
                                <li>
                                    <ul>{code}</ul>
                                </li>
                                <li>
                                    <ul>{product_name}</ul>
                                </li>
                                <li>
                                    <ul>$ {int(item[1][3])}</ul>
                                </li>
                                <li>
                                    <ul>{round(item[1][-1], 4)}</ul>
                                </li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
