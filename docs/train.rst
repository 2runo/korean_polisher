korean_polisher.train
=====================

.. contents::

korean_polisher.train.Transformer
---------------------------------

Transformer 모델

.. code-block:: Python

    Transformer(
        num_layers, d_model, num_heads, dff, input_vocab_size,
        target_vocab_size, pe_input, pe_target, rate=0.1
    )

Arguments
~~~~~~~~~

위에 있는 대로

Attributes
~~~~~~~~~~

- encoder - 인코더
- decoder - 디코더
- final_layer - 최종 레이어
- tk - 토크나이저
- learning_rate - 학습률
- optimizer - optimizer
- ckpt - 체크포인트
- ckpt_manager - 체크포인트 매니저

Methods
~~~~~~~

.. code-block:: Python

    Transformer.call(
        inp, tar, training,
        enc_padding_mask, look_ahead_mask, dec_padding_mask
    )

    Transformer.train_step(
        inp, tar
    )

    Transformer.predict(
        inp_sentences
    )

    Transformer.evaluate(
        inp, tar
    )

    Transformer.demo()

    Transformer.ckpt_save(
        epoch, batch_size
    )

    Transformer.history(
        test_loss, test_acc
    )

korean_polisher.train.train_loss
--------------------------------

train_loss

korean_polisher.train.train_accuracy
------------------------------------

train_accuracy

korean_polisher.train.get_model
-------------------------------

모델 반환
.. code-block:: Python

    get_model()
