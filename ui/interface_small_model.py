import gradio as gr
import polars as pl

all_dataframe = pl.read_csv("../data/recommends/dataframe_gradio.csv")

len_dataframe = all_dataframe.shape[0]
print(all_dataframe.shape[0])


def get_information(user_id, model):
    if model == "14M":
        return (
            all_dataframe[user_id][["video_category_history"]],
            all_dataframe[user_id][["author_id_history"]],
            all_dataframe[user_id][["categories_recommends"]],
            all_dataframe[user_id][["author_recommends"]],
            all_dataframe[user_id][["recommends"]],
        )
    elif model == "27M":
        return (
            all_dataframe[user_id][["video_category_history_27m"]],
            all_dataframe[user_id][["author_id_history_27m"]],
            all_dataframe[user_id][["categories_recommends_27m"]],
            all_dataframe[user_id][["author_recommends_27m"]],
            all_dataframe[user_id][["recommends_27m"]],
        )
    elif model == "87M":
        return (
            all_dataframe[user_id][["video_category_history_87m"]],
            all_dataframe[user_id][["author_id_history_87m"]],
            all_dataframe[user_id][["categories_recommends_87m"]],
            all_dataframe[user_id][["author_recommends_87m"]],
            all_dataframe[user_id][["recommends_87m"]],
        )


callback = gr.CSVLogger()

with gr.Blocks() as demo:
    gr.Markdown("# BERT4REC RECOMMENDS")
    gr.Markdown("The Bert is trained on a weekly history")
    user_name = gr.Textbox(placeholder="What is your name?", label="Name")
    with gr.Row():
        user_id = gr.Slider(
            1,
            len_dataframe,
            step=1,
            value=1,
            label="User id",
            info=f"Choose between 1 and {len_dataframe}",
        )
        model = gr.Dropdown(
            ["14M", "27M"],
            label="Model",
            info="Please choose one model!",
            value="27M",
        )
    with gr.Row():
        get_recommends_btn = gr.Button(value="Recommends", variant="primary")
    with gr.Row():
        history_categories = gr.DataFrame(
            label=f"Categories history for user", height=1500
        )
    with gr.Row():
        history_author = gr.DataFrame(label=f"Authors history for user", height=1500)
    with gr.Row():
        recommends_videos = gr.DataFrame(
            label=f"Recommends categories for user", height=1500
        )
    with gr.Row():
        recommends_author = gr.DataFrame(
            label=f"Recommends authors for user", height=1500
        )
    with gr.Row():
        recommends_ids = gr.DataFrame(label=f"Recommends IDS", height=1500)

    with gr.Row():
        answer = gr.Radio(
            ["Yes", "No", "IDK"],
            label="Choose",
            info="Do you like recommends?",
            value=None,
        )
    with gr.Row():
        save_btn = gr.Button("Сохранить выбор")

    callback.setup([user_name, user_id, model, answer], "./flagged_data_points")
    get_recommends_btn.click(
        get_information,
        inputs=[user_id, model],
        outputs=[
            history_categories,
            history_author,
            recommends_videos,
            recommends_author,
            recommends_ids,
        ],
    )
    # get_recommends_btn.click(get_recommends, inputs=[user_id, model, show_all], outputs=recommends)
    save_btn.click(
        lambda *args: callback.flag(args),
        [user_name, user_id, model, answer],
        None,
        preprocess=False,
    )


if __name__ == "__main__":
    demo.launch(debug=True, server_name="0.0.0.0", server_port=7860)
