import os, json
import pandas as pd

# this finds our json files
path_to_json = "data/keep_json/"
json_files = [
    pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith(".json")
]

jsons_data = pd.DataFrame(columns=["text", "created_at", "modified_at"])

for index, js in enumerate(json_files):
    try:
        with open(os.path.join(path_to_json, js)) as json_file:
            json_text = json.load(json_file)

            # here you need to know the layout of your json and each json has to have
            # the same structure (obviously not the structure I have here)
            text = json_text["textContent"]
            created_at = json_text["createdTimestampUsec"]
            modified_at = json_text["userEditedTimestampUsec"]
            # here I push a list of data into a pandas DataFrame at row given by 'index'
            jsons_data.loc[index] = [text, created_at, modified_at]
            print(jsons_data)
    except KeyError:
        print("Empty Note:", json_text, "File is:", json_file)

sorted_notes = jsons_data.sort_values(by="created_at", ascending=False)
sorted_notes.to_csv("data/all_notes.csv", index=True)
