# LecSlides_370K
> ### Towards Comprehensive Lecture Slides Understanding: Large-scale Dataset and Effective Method
>
> #### Enming Zhang, Yuzhe Li, Yuliang Liu, Yingying Zhu, Xiang Bai*
>
> <sup>* Corresponding Author

Towards Comprehensive Lecture Slides Understanding: Large-scale Dataset and Effective Method

## Features

* Establish a large-scale and multi-task dataset for comprehensive slides understanding, named LecSlides-370K.  Specifically, we collect 25,542 lectures composed of 370,078 slides. These lectures span 15 lecture areas, including business, natural sciences, computer sciences, and so on, significantly surpassing previous datasets in both scale and diversity.
* To extract the complex text relations and enhance slides understanding, we propose a novel method called SlideParser.  SlideParser outperforms all the related methods in both LecSlides-370K and SlideVQA datasets.



## Updates

- [x] Release dataset
- [x] Release evaluation code
- [x] Release training code

## Getting Started

You can download Slide370K dataset in [huggingface](https://huggingface.co/datasets/yzlii/LecSlides-370K/tree/main)

Annotation format

```
[
    {
        "image": List[str],
        "summary": str,
        "qa_pair": {
                    question: str,
                    answer: str,
                    explain: str,
                    type: str,
                    answer_length: int
                    },

    }
    
    ...
]
```

