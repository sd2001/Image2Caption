import modules as m

def define_model(vocab_size, max_length):
    inputs1=m.Input(shape=(4096,))
    fe1=m.Dropout(0.5)(inputs1)
    fe2=m.Dense(256, activation='relu')(fe1)
    inputs2=m.Input(shape=(max_length,))
    se1=m.Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2=m.Dropout(0.5)(se1)
    se3=m.LSTM(256)(se2)
    decoder1=m.add([fe2, se3])
    decoder2=m.Dense(256, activation='relu')(decoder1)
    outputs=m.Dense(vocab_size, activation='softmax')(decoder2)
    model=m.Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    m.plot_model(model, to_file='model.png', show_shapes=True)
    return model