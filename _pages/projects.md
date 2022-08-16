---
title: My projects - Zakarya
permalink: /projects
layout: splash
header:
  image: /assets/images/1930875.jpg
  overlay_filter: 0.5
---

Some of the project I worked on :

---

# 🚀 T5 Transformer meets French text for Punctuation ✨ 

👉 Try out my implementation : TextPunctuator a simple library for text punctuation based on a T5 model (Currently supports only French text).

👉 Ensuring proper punctuation is a key pre-processing step towards applying complex natural language processing algorithms. This is especially significant for textual sources where punctuation and casing are missing, such as the raw output of automatic speech recognition systems. Additionally, short text messages and online reviews on shopping platforms offer unreliable and often wrong punctuation and casing.

👉 Most of the existing approaches frames the problem as a Token Classification, So I decided to go with a Text generation approach instead.

👉 I fine-tuned the t5-base model on around 500k of high quality french texts.

👉 More languages will be supported soon !

You can now use the model on your own pipeline in only 3 lines of code 🤯.
<!-- ![](/assets/images/Snap.png) -->
![full](/assets/images/Snap.png){: .full}
Check out the:
[Model on Hugging Face 🤗](https://huggingface.co/ZakaryaRouzki/t5-punctuation){: .btn .btn--info}
[Github Repository](https://github.com/rouzki/TextPunctuator){: .btn .btn--info}



---


# 🚀 Multi Label Classification of web pages

The goal of the project is to produce a multi-label classifier capable of inferring the categories of a url, From a page URL we will try to predict in which categories fall this specific page and without scraping textes from the page (only using url text).

[Github Repository](https://github.com/rouzki/URL-NLP){: .btn .btn--info}

---

more projects on my profile here : [Github Repository](https://github.com/rouzki){: .btn .btn--info}