from compressor.sentence_splitter import split_sentences


text = """
Renewable energy production increased significantly between 2020 and 2025.

Solar generation increased by 42% during this period.

Government incentives and lower solar panel costs contributed to this growth.

It is important to note that renewable energy is becoming increasingly important.
"""


sentences = split_sentences(text)


print("\n===== SENTENCE SPLITTER TEST =====\n")

for index, sentence in enumerate(sentences, start=1):
    print(f"{index}. {sentence}")

print("\n=================================\n")
print(f"Total sentences: {len(sentences)}")