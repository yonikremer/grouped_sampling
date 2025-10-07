## 1. Change the benchmarking code to use grouped sampling and not vllm - DONE

## 2. Change the implementation of top p, top k and the sampling itself to the one by flashinfer - DONE

## 3. Compare the performance of the library VS vllm - DONE
not so good, vllm is so fast it is hard to compete with it lol 

## 4. Solving performance issues:

- tokenization—DONE
- creating causal masks-TODO, remember to cache the masks,
maybe even we can create one giant mask at initiation and use parts of it in inference time
- Memory usage-TODO (will allow us to increase batch size)
- Quantization by default-TODO
- Import and start-up time-TODO (mostly improve my dev-time)
- torch.compile by default

## 5.Test the quality of the output text in a professional manner (big task)
- Try to use https://github.com/EleutherAI/lm-evaluation-harness/tree/main

## 6. Rethink the API of the library (lower priority)

## 7. new features (lower priority)
