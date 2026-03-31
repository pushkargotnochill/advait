from transformers import pipeline


sentiment_report = pipeline("sentiment-analysis")
result = sentiment_report("i love everyone")
print(result)


summarizer = pipeline( model="facebook/bart-large-cnn")
text = """Because Ichigo lacks a traditional, proactive goal, critics sometimes label him as a passive character who only moves when the plot demands it. Yet, this is precisely what makes him so compelling. Ichigo is a normal teenager thrust into extraordinary, terrifying circumstances. He values his normal life, his schoolwork, and his friends above the politics of the supernatural Soul Society.

When he loses his powers after defeating the central villain Sosuke Aizen, he falls into a realistic, profound depression. This highlights that his powers were never just weapons to him; they were his insurance against powerlessness. Watching him claw his way back from despair repeatedly proves that his greatest strength is not his massive spiritual energy, but his sheer, unyielding resolve.
"""
summary = summarizer(text, max_length =30, min_length = 20, do_sample = False)

print(summary[0]['summary_text'])



qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

context_1 = """
Because Ichigo lacks a traditional, proactive goal, critics have occasionally 
accused him of being a bland protagonist. When he loses his powers after 
defeating the central villain Sosuke Aizen, Ichigo trains to recover them 
using a method called the Fullbring.
"""

result = qa(question="How does Ichigo recover his powers?", context=context_1)
print(result['answer'])

#explaination : facebook/bart-large-cnn is a pre trained model made by  meta , is effective in generating summaries of long text , for natulral language generation , it combines with models like BERT and GPT
#explaination2 = deepset/roberta-base-squad2 is a popular NLP model that specialises in extractive question answering , The model identifies and extracts the exact span of text from a given context that answers a specific question.
