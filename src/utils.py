import os.path
import sys

def query_boolean(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    Source: http://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    valid = {'yes': True, 'y': True, "ye": True, 'j': True, 'ja': True,
             "no": False, "n": False, 'nein': False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').")


def query_overwrite(path):
   if not os.path.exists(path):
       return True
   question = ('File {} does exist.\n'
               'Overwrite it?'.format(path))
   return query_boolean(question, default='yes')


def touch(path, clear=False):
   os.makedirs(os.path.dirname(path), exist_ok=True)
   if not os.path.isdir(path):
       open(path, 'a').close()
       if clear:
           open(path, 'w').close()
