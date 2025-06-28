import argparse

class ParseKVAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        assert values is not None
        for each in values:
            try:
                key, value = each.split("=")
                getattr(namespace, self.dest)[key] = value
            except ValueError as ex:
                message = "\nTraceback: {}".format(ex)
                message += "\nError on '{}' || It should be 'key=value'".format(each)
                raise argparse.ArgumentError(self, str(message))