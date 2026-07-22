import csv
import html
import json
import re
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Parser(HTMLParser):
 def __init__(self): super().__init__(); self.parts=[]
 def handle_data(self,data): self.parts.append(data)
def text(value):
 p=Parser(); p.feed(html.unescape(value)); return re.sub(r"\n{3,}","\n\n",re.sub(r"[ \t]+"," ","\n".join(p.parts))).strip()
def condition(value):
 if isinstance(value,dict):
  if value.get("LeftOperand")=="condition" and value.get("Operator")=="EqualTo": return value["RightOperand"]
  for child in value.values():
   found=condition(child)
   if found:return found
 if isinstance(value,list):
  for child in value:
   found=condition(child)
   if found:return found
 return None
survey=json.loads((ROOT/"survey/survey.json").read_text())["result"]
blocks={block["ID"]:block for block in survey["Blocks"].values()}
ids=defaultdict(list)
def walk(flow,code=None):
 for node in flow:
  current=condition(node.get("BranchLogic")) or code
  if current and node.get("Type") in {"Block","Standard"} and node.get("ID") in blocks:
   ids[current]+=[x["QuestionID"] for x in blocks[node["ID"]].get("BlockElements",[]) if x.get("Type")=="Question"]
  walk(node.get("Flow",[]),current)
walk(survey["SurveyFlow"]["Flow"])
with (ROOT/"survey/condition_codenames.csv").open(newline="") as f: names=list(csv.DictReader(f))
out=defaultdict(list)
for row in names:
 qs=[survey["Questions"][qid] for qid in dict.fromkeys(ids[row["code_name"]]) if survey["Questions"][qid].get("QuestionType")=="DB"]
 if not qs: raise ValueError(f"No display text for {row['code_name']!r}")
 out[row["title"]].append(text(qs[0]["QuestionText"]))
out={key:out[key] for key in sorted(out,key=str.casefold)}
(ROOT/"qstn_data/conditions.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
print(f"Wrote {sum(map(len,out.values()))} variants across {len(out)} conditions.")
