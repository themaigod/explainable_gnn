class HINData:
    keys = [
        'graph',
        'node_types',
        'node_features',
        'labels',
        'subgraph',
        'meta_path',
    ]

    def __init__(self, **kargs):
        for k in self.keys:
            if k in kargs:
                setattr(self, k, kargs[k])

            if k + '_processor' in kargs:
                setattr(self, k + '_processor', kargs[k + '_processor'])

            if k + '_processor_kargs' in kargs:
                setattr(self, k + '_processor_kargs', kargs[k + '_processor_kargs'])

        if kargs.get('unified', False):
            self.unified_data()

    def unified_data(self):
        for k in self.keys:
            if getattr(self, k, None) is not None and getattr(self, k + '_processor',
                                                              None) is not None:
                if getattr(self, k + '_processor_kargs', None) is not None:
                    setattr(self, k, getattr(self, k + '_processor')(getattr(self, k),
                                                                     **getattr(self,
                                                                               k + '_processor_kargs')))
                else:
                    setattr(self, k, getattr(self, k + '_processor')(getattr(self, k)))

    def set_processor(self, processor_target, processor):
        processor_name = processor_target + '_processor'
        setattr(self, processor_name, processor)

    def return_data(self):
        return {
            k: getattr(self, k) for k in self.keys if
            getattr(self, k, None) is not None
        }

    def return_all(self):
        data = {
            k: getattr(self, k) for k in self.keys
        }
        for k in self.keys:
            if k + '_processor' in data:
                data[k + '_processor'] = getattr(self, k + '_processor')
            if k + '_processor_kargs' in data:
                data[k + '_processor_kargs'] = getattr(self, k + '_processor_kargs')
        return data

    def check_data(self):
        pass
