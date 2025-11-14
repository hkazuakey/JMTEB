{
  jqara: {
    class_path: 'RerankingEvaluator',
    init_args: {
      val_query_dataset: {
        class_path: 'HfRerankingQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'jqara-query',
        },
      },
      test_query_dataset: {
        class_path: 'HfRerankingQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'jqara-query',
        },
      },
      doc_dataset: {
        class_path: 'HfRerankingDocDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'corpus',
          name: 'jqara-corpus',
        },
      },
    },
  },
}
