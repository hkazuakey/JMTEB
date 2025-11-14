{
  mldr_reranking: {
    class_path: 'RerankingEvaluator',
    init_args: {
      val_query_dataset: {
        class_path: 'HfRerankingQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'mldr-reranking-query',
        },
      },
      test_query_dataset: {
        class_path: 'HfRerankingQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'mldr-reranking-query',
        },
      },
      doc_dataset: {
        class_path: 'HfRerankingDocDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'corpus',
          name: 'mldr-reranking-corpus',
        },
      },
    },
  },
}
