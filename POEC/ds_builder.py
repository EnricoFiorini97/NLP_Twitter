import pandas as pd
import spacy

from pandas.core.indexes import category


def hits(s="", lst=[]) -> int:
        total_hits = 0
        for token in lst:
                if token in s:
                        total_hits += 1

        return total_hits

def hits_ent(ent=[], lst=[]) -> int:
        total_ent_hits = 0      
        for e in ent:
                if e in lst:
                        total_ent_hits += 1
        return total_ent_hits

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

        categories_ents = {
                'Artworks':     ['WORK_OF_ART', 'TIME', 'QUANTITY', 'CARDINAL', 'PRODUCT','PERCENT', 'EVENT', 'LOC', 'GPE'], 
                'Festivities':  ['TIME', 'QUANTITY', 'PRODUCT', 'MONEY', 'PERCENT', 'DATE', 'EVENT', 'GPE'], 
                'History':      ['WORK_OF_ART', 'TIME', 'CARDINAL', 'PRODUCT', 'LAW', 'DATE', 'EVENT', 'LOC', 'GPE'], 
                'OnThisDay':    ['TIME', 'LOC', 'PRODUCT', 'MONEY', 'PERCENT', 'DATE', 'EVENT', 'QUANTITY', 'GPE'], 
                'Promotions':   ['TIME', 'PRODUCT', 'LOC', 'MONEY', 'DATE', 'EVENT', 'QUANTITY', 'GPE', 'WORK_OF_ART'], 
                'VIP_CIT':      ['TIME', 'QUANTITY', 'LAW', 'CARDINAL', 'PRODUCT', 'PERCENT', 'DATE', 'LOC', 'GPE']
        }

        other = 0
        NLP = spacy.load("en_core_web_sm")
        df = pd.DataFrame(columns = ["F_NAME", "F_CONTENT", "CATEGORY"])
        with open("/home/enrico/Documents/repos/NLP_Twitter/text_only_rawdataset/merge.txt", "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                        content = line.strip()
                        max_cat_score = 0.0
                        max_score_cat_label = ""
                        for category, tokens in categories.items():
                                curr_score = hits(content, tokens) / len(categories[category]) #normalizzazione
                                content_ent = NLP(content)
                                entities_tweet = [entity.label_ for entity in content_ent.ents]
                                curr_score_ent = hits_ent(entities_tweet, categories_ents[category]) / len(categories_ents[category])
                                curr_score += curr_score_ent
                                if curr_score > max_cat_score:
                                        max_score_cat_label = category
                                        max_cat_score = curr_score      
                        if max_cat_score > 0.0:
                                cat_hits[max_score_cat_label] += 1
                                df.loc[len(df)] = [f"{max_score_cat_label}-{cat_hits[max_score_cat_label]}.txt", content, max_score_cat_label]
                                max_cat_score = 0
                                #content_ent = NLP(content)
                                '''for entity in content_ent.ents:
                                        categories_ents[max_score_cat_label].add(entity.label_)'''
                        else:
                                df.loc[len(df)] = [f"Other-{other}.txt", content, "Other"]
                                other += 1

        df.to_csv("dataset_2_ent.csv")   
        print(categories_ents)  

if __name__ == "__main__":
        main() 
