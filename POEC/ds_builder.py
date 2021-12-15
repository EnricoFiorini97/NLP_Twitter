import pandas as pd
from pandas.core.indexes import category


def hits(s="", lst=[]) -> int:
        total_hits = 0
        for token in lst:
                if token in s:
                        total_hits += 1

        return total_hits

def main() -> None:
        categories = {
                        "Artworks" :    ["cup", "photo", "painter", "opera", "ceramic", "portrait", "paint" ,"japan", "august", "vase", "photo"], 
                        "Festivities" : ["staff", "celebrate", "birthday","anniversary", "happy", "happy birthday", "celebrating", "wish", "fun"], 
                        "History" :     ["history", "war", "past", "years", "century", "antinous", "food", "artistinterventions", "historian tashamarks", "curator blog", "historian"], 
                        "OnThisDay" :   ["today", "onthisday", "tonight", "born", "die" , "happen", "happens today", "happened today"], 
                        "Promotions" :  ["tickets", "check", "check out", "week", "exhibition", "explore", "2021", "2020", "open", "opens", "don't miss"], 
                        "VIP_CIT" :     ["artist", "said", "phrase", "citation", "famous", "series", "artquote", "architect", "photographer", "find inspire"]
                     }

        cat_hits = {
                        "Artworks" :    0, 
                        "Festivities" : 0, 
                        "History" :     0, 
                        "OnThisDay" :   0, 
                        "Promotions" :  0, 
                        "VIP_CIT" :     0
                }

        other = 0
        df = pd.DataFrame(columns = ["F_NAME", "F_CONTENT", "CATEGORY"])
        with open("text_clean.txt", "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                        content = line.strip()
                        max_cat_score = 0.0
                        max_score_cat_label = ""
                        for category, tokens in categories.items():
                                curr_score = hits(content, tokens) / len(categories[category]) #normalizzazione
                                if curr_score > max_cat_score:
                                        max_score_cat_label = category
                                        max_cat_score = curr_score      
                        if max_cat_score > 0.0:
                                cat_hits[max_score_cat_label] += 1
                                df.loc[len(df)] = [f"{max_score_cat_label}-{cat_hits[max_score_cat_label]}.txt", content, max_score_cat_label]
                                max_cat_score = 0
                        else:
                                df.loc[len(df)] = [f"Other-{other}.txt", content, "Other"]
                                other += 1

        df.to_csv("dataset_2.csv")     

if __name__ == "__main__":
        main() 
