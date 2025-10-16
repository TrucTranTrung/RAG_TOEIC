\documentclass[border=6pt]{standalone}
\usepackage[edges]{forest}
\forestset{
  default preamble={
    for tree={
      draw,
      rounded corners,
      align=center,
      inner sep=3pt,
      l sep+=6pt,
      s sep+=6pt,
      edge={-latex},
      font=\small
    }
  },
  group/.style={fill=yellow!25},
  input/.style={fill=red!20},
  proc/.style={fill=blue!15},
  tool/.style={fill=orange!30},
  aux/.style={fill=green!20},
}

\begin{document}
\begin{forest}
for tree={parent anchor=south, child anchor=north}
[Host / Tools / Dumb Question, group
  [Inputs, input
    [Images]
    [Image \& text/voice]
    [Text]
    [Speech (raw audio)]
  ]
  [Listening / Speech path, tool
    [Speech to text, tool
      [Lis2+4 / Prompt, tool]
      [Lis3, tool
        [Tools check gender, tool]
      ]
    ]
  ]
  [Reading path, proc
    [Reading 5, proc
      [*Tool grammar/Vocab/Collocation, aux]
    ]
    [Reading 6 / Prompt, proc
      [*Tool grammar/Vocab/Collocation, aux]
    ]
    [Reading 7 (Context Q?), proc
      [Summarize tools \\ Extract information tools, aux]
    ]
  ]
  [Vision path, proc
    [LLM/OCR $\rightarrow$ Text in Image, aux]
    [Listening part 1, proc]
    [Image-first tool $\rightarrow$ LLM $\rightarrow$ Text Image, aux]
  ]
  [RAG / Embedding pipeline, proc
    [Question + passage (Tìm câu \& tách), proc]
    [Chunking passage, proc
      [Traditional]
      [Semantic]
    ]
    [Embedding (nomic-embed-text), proc]
    [Cosine similarity, proc]
    [BOT, proc]
  ]
]
\end{forest}
\end{document}
