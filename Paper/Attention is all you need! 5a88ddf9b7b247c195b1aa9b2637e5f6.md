# Attention is all you need!

# Introduction

기존 sequential 모델들은 메모리 제약 때문에 본질적으로 병렬화를 배제하여 batch 문제가 있어 긴 길이의 sequence에 취약하다.

어텐션 메커니즘은 input, output의 거리에 관계 없이 dependencies 모델링을 할 수 있다.

하지만 대부분 recurrent network와 함께 사용 돼 효율적인 병렬화가 불가능했다.

본 연구는 **recurrence를 제거하고 input과 output 사이의 global 의존성을 학습하기 위한 attention 메커니즘만을 사용한 모델 transformer를 제안**한다. Transformer는 8개의 P100 GPU로 12시간 학습하여 **병렬화 및 SOTA를 달성**한다.

그래서 The transformer를 제안한다!

This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples

→ 메모리 제약이 예제 전체에서 batching을 제한하기 때문에, 본질적으로 sequential nature는 병렬화를 배제하여 긴 sequence 길이에 취약하다. 

The fundamental constraint of sequential computation remains.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, Allowing modeling of dependencies without regard to their distance in the input or output sequences

→ 어텐션 메커니즘은 sequence modeling, transduction models 등에서 필수적인 요소가 됐으며, input과 output의 거리에 관계 없이 의존성의 모델링을 할 수 있게 한다.

In this work we propose the Transformer.

a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output

# Background

sequential 연산을 줄이기 위한 모델들이 있었으나, 이는 cnn을 사용하여 원거리 같이 의존성을 학습하는 것이 어렵다.

In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

→ ???

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence

→ Self-attention (intra-attention)은 sequence의 representation을 계산하기 위해 single sequence의 다른 위치를 연관시킨다 (?)

end-to end memory net이 뭐지?

end-to-end memory network는 sequence-aligned recurrence 대신 recurrent attention 메커니즘을 기반으로 하여 single-language QnA와 언어 모델링 tasks에 좋은 성능을 보인다.

transduction? 형질도입..? [https://en.wikipedia.org/wiki/Transduction_(machine_learning)](https://en.wikipedia.org/wiki/Transduction_(machine_learning))

Transformer는 self-attention만 사용하여 input과 output의 representation을 계산하는 transduction model이다.

without using sequence-aligned RNNs or Convolution.

# Model Architecture

대부분 모델의 경우?

At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.

각 step마다 model은 auto-regressive, 이전에 생성된 output을 추가적인 input으로 다음 단계에서 사용한다. ( → 이전 단계가 완료돼야 다음 단계도 이어짐 / 순차적으로 진행 / 병렬적 처리 불가)

![Untitled](Attention%20is%20all%20you%20need!%205a88ddf9b7b247c195b1aa9b2637e5f6/Untitled.png)

## 3.1 Encoder and Decoder Stacks

- Encoder

encoder는 N=6 동일한 layers들의 stack으로 구성된다.

첫번째는 multi-head self-attention mechanism, 두번째는 simple, position-wise fully connected feed-forward network.

2 sub-layer마다 residual connection와 layer normalization을 사용한다.

즉 sub-layer의 output은 (LayerNorm(x+Sublayer(x))

residual connection을 용이하게 하기 위해 embedding layer을 포함한 모든 sub-layers 들은 512-dim을 가진 output을 만든다.

→ N=6 layers들의 stack
    layer = layerNorm(x+Multi-head self-attention(x)) + LayerNorm(x+FFN(x))

### Decoder

N=6 동일한 layers들의 stack으로 구성

decoder는 3rd sub-layer를 도입 (multi-head attention, encoder의 output에 대해 수행)

encoder와 동일하게 2 sub-layer마다 residual connection과 layer normalization 수행

self-attention sub-layer를 수정하여 position이 subsequent positions에 attending하는 것을 막는다(?).. → masking

Output embedding이 한 위치의 offset이라는 사실과 결합한 masking은 position i에 대한 예측이 i보다 적은 수의 위치 (i 이전에 output된 위치)에서 알려진 output에만 의존할 수 있도록 한다 **(미래 시점을 막아주는 역할)**. ..?

This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

→ N=6 layers들의 stack
    layer = layerNorm(x+Masked Multi-head self-attention(x)) + layerNorm(x+Multi-head self-attention(x)) + LayerNorm(x+FFN(x))

## 3.2 Attention

attention 함수는 query와 key-value 쌍을 query, keys, values가 모두 벡터인 output으로 mapping 하는 것으로 설명할 수 있다.

output은 weighted sum of the values로 계산이 된다. (각 value의 weight는 해당되는 키와 query의 compatibility function에 의해 계산된다.)

![Untitled](Attention%20is%20all%20you%20need!%205a88ddf9b7b247c195b1aa9b2637e5f6/Untitled%201.png)

- Q : 영향을 받는 벡터 (query)
- K : 영향을 주는 벡터 (key)
- V : 주는 영향의 가중치 벡터 (value)

### 3.2.1 Scaled Dot-Product Attention

input - queries and keys of dim $d_k$, values of dim $d_v$ 3가지로 이루어짐

query와 모든 key에 대해 dot product.

→ divide eatch by $\sqrt{d_k}$

→ (Mask option..?)

→ softmax function to obtain the weights on the values

⇒ $Attention(Q,K,V) = softmax(QK^T/\sqrt{d_k})V$

일반적으로 자주 사용하는 attention 함수는 additive attention, dot-product attention(multiplicative)

dot-product의 경우 scaling factor를 제외하면, 위의 알고리즘과 동일하다

additive attention의 경우 single hidden layer인 feed-forward network를 사용하여 compatibility function을 계산한다.

이론적으론 complexity가 동일하지만, dot-product의 경우가 실제로 더 빠름. 공간적으로도 효율적

그래서 dot 자주 씀 ㅇㅇ

$d_k$가 작으면 동일하게 작동하나, 클 수록 additive가 outperform!

$d_k$의 큰 값에 대해 dot products는 크기가 크게 증가하여 softmax 함수를 small gradients를 갖게 함.

→ 이를 막기 위해, $1\over{\sqrt{d_k}}$만큼 scale.

### 3.2.2 Multi-Head Attention

$d_{model}$-dim의 keys, values, queries 해당하는 a single attention function을 사용하는 것보다

queries, keys, values를 각각 $d_k, d_k, d_v$차원에 대해 학습된 서로 다른 linear projection을 사용하여 h번 linear projection하는 것이 유익하다는 것을 발견했다.

각각의 project된 version에 대해서 attention function을 병렬적으로 적용, $d_v$-dim output values를 산출.

→ 이들을 concatenate하고 한 번 더 projected하여 결과를 얻어냄.

Multi-head attention을 통해 다른 position의 서로 다른 representation subspaces로부터의 정보를 공동으로 attend (?)할 수 있다.

⇒ $MultiHead(Q,K,V) = Concat(head1,...,head_h)W^o$

$where,\ head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$

### 3.2.3 Applications of Attention in our Model

The transformer uses multi-head attention in three different ways:

- “encoder-decoder attention” layer의 경우, query는 이전 decoder layer에서 key와 value는 encoder의 output에서 얻는다.
    
    이는 decoder의 모든 position이 input sequence의 모든 위치를 처리할 수 있다..?
    
- encoder는 self-attention layer을 포함한다.
self-attention layer의 모든 key, value, query들은 똑같은 위치(해당 경우, encoder의 이전 layer의 output)에서 가져온다.
    
    encoder의 각 위치는 encoder 이전 layer의 모든 위치를 attend 할 수 있다!
    
- decoder의 self-attention layer는 각 position가 해당 postion을 포함하여 모든 position을 attend할 수 있게 한다.
    
    auto-regressive property를 지키기 위해 decoder에서 leftward information flow를 막아야 한다.(미래 시점 단어를 볼 수 없도록)
    
    → 이를 위해 scaled dot-product attention의 내부에서 softmax에서 illegal한 연결의 input들을 masking out! (setting to $-\infty$)
    

### 3.3 Position-wise Feed-Forward Networks

attention sub-layer외에도 encoder, decoder 각각에 fully connected feed-forward network가 존재하며 이는 각 포지션마다 개별적으로 동일하게 적용된다.

2개의 선형 transformations 사이에 ReLU가 적용된다.

⇒ $FFN(x) = max(0,xW_1+b_1)W_2+b_2$서

선형 transformation은 다른 위치에선 동일하지만, layer마다 다른 parameter를 사용한다.

이를 설명하는 다른 방법은 kernel size 1인 2개의 convolution을 사용하는 것이다.

input 과 output 은 $d_{model} = 512$ 차원을 가지고 있으며 inner-layer는 $d_{ff} = 2048$ 차원을 갖는다.

### 3.4 Embeddings and Softmax

다른 sequence transduction models과 동일하게, input token과 output token들을 $d_{model}$차원의 vector로 변환하기 위해 학습된 embedding을 사용한다

decoder output을 predicted next-token probability로 변환하기 위해 학습된 linear transformation과 softmax를 사용한다.

2개의 embedding layer와 softmax 이전 linear transformation에서 동일한 weight matrix를 공유한다.

### 3.5 Positional Encoding

해당 모델은 recurrence와 convolution이 없어, order of the sequence를 사용하려면

sequence에서 token의 상대적 혹은 절대적인 포지션을 추가해야한다.

encoder와 decoder stack의 bottoms, input embedding에 ‘positional encodings’을 추가

positional encodings은 동일한 $d_{model}$차원을 가진다.

learned, fixed 등 많은 선택이 있음.

$PE_{(pos, 2i)}=sin(pos/10000^{2i/d_{model}})$

$PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})$

(pos는 postion, i는 dimension 해당)

## 4 Why Self-Attention

self-attention 쓰는 이유?

1. layer별 총 computational complexity
2. amount of computation (필요한 순차 연산들의 최소 수로 측정, 병렬 처리 할 수 있는)
3. network에서 장거리 의존성 사이 path length.