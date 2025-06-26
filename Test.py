import textstat

text = "The Australian platypus is seemingly a hybrid of a mammal and reptilian creature.how I distinguashed the situation call numb."

print("Flesch Reading Ease:", textstat.flesch_reading_ease(text))
print("SMOG Index:", textstat.smog_index(text))
print("Difficult Words:", textstat.difficult_words(text))
print("Lexicon Count:", textstat.lexicon_count(text))
print("Sentence Count:", textstat.sentence_count(text))
