import sys
import traceback
import pkgutil

print('sys.path[0:3]=', sys.path[0:3])

try:
    __import__('analise_qualidade_vinhos')
    print('IMPORT_OK')
except Exception as e:
    print('IMPORT_ERROR', e)
    traceback.print_exc()

print('modules in src:', [m.name for m in pkgutil.iter_modules(['src'])])
