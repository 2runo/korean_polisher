korean_polisher.train
=====================

.. contents::

korean_polisher.train.CustomSchedule
------------------------------------

학습을 위한 스케줄 클래스(Adam의 인자로만 사용)

.. code-block:: Python

    CustomSchedule(
        d_model
    )

korean_polisher.train.Transformer
---------------------------------

Transformer 모델

.. code-block:: Python

    Transformer(
        num_layers, d_model, num_heads, dff, input_vocab_size,
        target_vocab_size, pe_input, pe_target, rate=0.1
    )

korean_polisher.train.임포트 가능한 것들
---------------------------------

loss_function,
create_masks,
create_padding_mask,
create_look_ahead_mask,
train_loss, train_accuracy, train_step_signature

korean_polisher.train.demo
--------------------------------

.. code-block:: Python

    demo()

korean_polisher.train.train_step
--------------------------------

.. code-block:: Python

    train_step(
        model, inp, tar
    )

korean_polisher.train.ckpt_save
--------------------------------

.. code-block:: Python

    ckpt_save(
        epoch, batch_iter
    )

korean_polisher.train.history
--------------------------------

.. code-block:: Python

    history(
        test_loss, test_acc
    )

korean_polisher.train.evaluate
--------------------------------

.. code-block:: Python

    evaluate(
        model: Transformer, inp, tar
    )

korean_polisher.train.tk
--------------------------------

토크나이저
