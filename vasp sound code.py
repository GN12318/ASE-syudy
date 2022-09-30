# Copyright (C) 2008 CSC - Scientific Computing Ltd.
"""This module defines an ASE interface to VASP.

Developed on the basis of modules by Jussi Enkovaara and John
Kitchin.  The path of the directory containing the pseudopotential
directories (potpaw,potpaw_GGA, potpaw_PBE, ...) should be set
by the environmental flag $VASP_PP_PATH.

The user should also set the environmental flag $VASP_SCRIPT pointing
to a python script looking something like::

   import os
   exitcode = os.system('vasp')

Alternatively, user can set the environmental flag $VASP_COMMAND pointing
to the command use the launch vasp e.g. 'vasp' or 'mpirun -n 16 vasp'

http://cms.mpi.univie.ac.at/vasp/
"""

import os     #operating system对文件，文件夹执行操作的一个模块
import sys    #sys模块是与python解释器交互的一个接口。sys 模块提供了许多函数和变量来处理 Python 运行时环境的不同部分
import re     #re模块是python提供的一套关于处理正则表达式的模块.
import numpy as np # Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库
import subprocess   #fork子进程
from contextlib import contextmanager  #contextlib模块的作用是提供更易用的上下文管理器。 它提供了3个对象：装饰器 contextmanager、函数 nested 和上下文管理器 closing
from pathlib import Path #提供表示文件系统路径的类
from warnings import warn   #模块抑制来自同一来源的重复消息，以减少一遍又一遍地看到相同警告的烦恼
from typing import Dict, Any    #类型检查
from xml.etree import ElementTree   #xml.etree.ElementTree模块实现了一个简单而高效的API用于解析和创建XML数据。

import ase
from ase.io import read, jsonio  #文件读写
from ase.utils import PurePath  #util 是一个 Node.js 核心模块，提供常用函数的集合，用于弥补核心 JavaScript 的功能过于精简的不足。
from ase.calculators import calculator   #计算模块、函数
from ase.calculators.calculator import Calculator   #Calculator类
from ase.calculators.singlepoint import SinglePointDFTCalculator   #单点计算？
from ase.calculators.vasp.create_input import GenerateVaspInput     #vasp输入生成器，如incar


class Vasp(GenerateVaspInput, Calculator):  # type: ignore    
    """ASE interface for the Vienna Ab initio Simulation Package (VASP),    #接口，各参数特征
    with the Calculator interface.  #计算界面

        Parameters:  #参数

            atoms:  object  #对象
                Attach an atoms object to the calculator. #将原子加入计算器

            label: str  #标签：字符串
                Prefix for the output file, and sets the working directory. #输出文件前缀，设置工作目录
                Default is 'vasp'.

            directory: str   #目录，字符
                Set the working directory. Is prepended to ``label``.在标签的下拉菜单设置工作目录

            restart: str or bool    重启
                Sets a label for the directory to load files from. 设置要从中加载文件的目录标签
                if :code:`restart=True`, the working directory from    使用的是‘directory'中的工作目录
                ``directory`` is used.

            txt: bool, None, str or writable object
                - If txt is None, output stream will be supressed 如果txt为None，则输出流将被压制

                - If txt is '-' the output will be sent through stdout    #stdout标准输出？如果txt是'-'，输出将通过stdout发送

                - If txt is a string a file will be opened,\   如果txt是一个字符串，文件将被打开，
                    and the output will be sent to that file.  输出将被发送到该文件。

                - Finally, txt can also be a an output stream,\  最后，txt也可以是一个输出流，\
                    which has a 'write' attribute.  哪个具有“可写入”属性

                Default is 'vasp.out'

                - Examples:

                    >>> Vasp(label='mylabel', txt='vasp.out') # Redirect stdout 重新定向标准输出
                    >>> Vasp(txt='myfile.txt') # Redirect stdout
                    >>> Vasp(txt='-') # Print vasp output to stdout 将vasp输出打印到标准输出
                    >>> Vasp(txt=None)  # Suppress txt output    抑制三种输出

            command: str
                Custom instructions on how to execute VASP. Has priority over   关于如何执行VASP的自定义说明。优先级高于环境变量。
                environment variables.
    """
    name = 'vasp'
    ase_objtype = 'vasp_calculator'  # For JSON storage   对象类型 JSON储存

    # Environment commands     环境中设置的命令
    env_commands = ('ASE_VASP_COMMAND', 'VASP_COMMAND', 'VASP_SCRIPT')

    implemented_properties = [                                              #实现属性
        'energy', 'free_energy', 'forces', 'dipole', 'fermi', 'stress',     #能量，自由能，力，偶极子，费米，压力，原子磁矩？
        'magmom', 'magmoms'
    ]

    # Can be used later to set some ASE defaults   以后可以用来设置一些ASE默值吗
    default_parameters: Dict[str, Any] = {}       #默认参数  的类？ Dict

-------------------------------------------------------------------------------------------------------------------------------
    def __init__(self,                      #定义新类，写入默认值
                 atoms=None,
                 restart=None,
                 directory='.',
                 label='vasp',
                 ignore_bad_restart_file=Calculator._deprecated,   #忽视坏的重启文件？
                 command=None,
                 txt='vasp.out',         #只要运行就会默认生成txt='vasp.out'
                 **kwargs):

        self._atoms = None    #原子属性
        self.results = {}     #结果属性

        # Initialize parameter dictionaries    初始化参数字典
        GenerateVaspInput.__init__(self)
        self._store_param_state()  # Initialize an empty parameter state     初始化 储存参数 状态

        # Store calculator from vasprun.xml here - None => uninitialized  存储计算器从vasprune .xml这里- None =>未初始化？？？？
        self._xml_calc = None    #XML 是一种很像HTML的可扩展标记语言

        # Set directory and label   设置目录和标签
        self.directory = directory
        if '/' in label:
            warn(('Specifying directory in "label" is deprecated, '   #警告 '在"label"中指定目录已弃用，'用"directory"代替')
                  'use "directory" instead.'), np.VisibleDeprecationWarning)   #可视反对警告
            if self.directory != '.':
                raise ValueError('Directory redundantly specified though '    #指定目录冗余？
                                 'directory="{}" and label="{}".  '
                                 'Please omit "/" in label.'.format(            #请在标签中省略/
                                     self.directory, label))
            self.label = label
        else:
            self.prefix = label  # The label should only contain the prefix

        if isinstance(restart, bool):
            if restart is True:
                restart = self.label
            else:
                restart = None

        Calculator.__init__(                                #初始化Calculator类？
            self,
            restart=restart,
            ignore_bad_restart_file=ignore_bad_restart_file,     #忽视坏的重启文件？
            # We already, manually, created the label    我么已经手动创建了标签
            label=self.label,
            atoms=atoms,
            **kwargs)

        self.command = command

        self._txt = None
        self.txt = txt  # Set the output txt stream  设置输出文本流
        self.version = None

        # XXX: This seems to break restarting, unless we return first.  这似乎会中断重启，除非我们先返回。
        # Do we really still need to enfore this?   我们真的还需要加强这一点吗?

        #  # If no XC combination, GGA functional or POTCAR type is specified,   如果没有XC组合，则指定GGA功能类型或POTCAR类型，
        #  # default to PW91. This is mostly chosen for backwards compatibility.   默认的PW91。这样做主要是为了向后兼容。
        # if kwargs.get('xc', None):    如果关键词参数  没有xc
        #     pass
        # elif not (kwargs.get('gga', None) or kwargs.get('pp', None)):
        #     self.input_params.update({'xc': 'PW91'})   gga和pp也没有的话，更新成pw91
        # # A null value of xc is permitted; custom recipes can be   xc可空值
        # # used by explicitly setting the pseudopotential set and   自定义方法可以通过显式设置赝势集和来使用
        # # INCAR keys
        # else:
        #     self.input_params.update({'xc': None})
--------------------------------------------------------------------------------------------------------------

    def make_command(self, command=None):
        """Return command if one is passed, otherwise try to find
        ASE_VASP_COMMAND, VASP_COMMAND or VASP_SCRIPT.
        If none are set, a CalculatorSetupError is raised"""
        if command:
            cmd = command
        else:
            # Search for the environment commands
            for env in self.env_commands:
                if env in os.environ:
                    cmd = os.environ[env].replace('PREFIX', self.prefix)
                    if env == 'VASP_SCRIPT':
                        # Make the system python exe run $VASP_SCRIPT
                        exe = sys.executable
                        cmd = ' '.join([exe, cmd])
                    break
            else:
                msg = ('Please set either command in calculator'
                       ' or one of the following environment '
                       'variables (prioritized as follows): {}').format(
                           ', '.join(self.env_commands))
                raise calculator.CalculatorSetupError(msg)
        return cmd

-------------------------------------------------------------------------------------------------------------------------
    def set(self, **kwargs):
        """Override the set function, to test for changes in the     覆盖set函数
        Vasp Calculator, then call the create_input.set()            检查Vasp计算器中的更改，然后调用create_input.set() 
        on remaining inputs for VASP specific keys.                  VASP特定键的剩余输入。

        Allows for setting ``label``, ``directory`` and ``txt``        允许设置' '标签' '，' '目录' '和' ' txt ' '而不重置计算器的结果。 
        without resetting the results in the calculator.
        """
        changed_parameters = {}                 #改变参数

        if 'label' in kwargs:
            self.label = kwargs.pop('label')

        if 'directory' in kwargs:
            # str() call to deal with pathlib objects   调用Str()来处理库路径对象
            self.directory = str(kwargs.pop('directory'))

        if 'txt' in kwargs:
            self.txt = kwargs.pop('txt')

        if 'atoms' in kwargs:
            atoms = kwargs.pop('atoms')
            self.atoms = atoms  # Resets results   重置的结果，如果这一系列的词出现在关键词 参数中，赋值？

        if 'command' in kwargs:
            self.command = kwargs.pop('command')     #Pop这函数在python中的主要作用就是对元素进行删除的操作

        changed_parameters.update(Calculator.set(self, **kwargs))

        # We might at some point add more to changed parameters, or use it  在某些时候，我们可能会向更改的参数添加更多内容，或者使用它
        if changed_parameters:
            self.clear_results()  # We don't want to clear atoms
        if kwargs:
            # If we make any changes to Vasp input, we always reset  如果我们对Vasp输入做了任何更改，我们总是重置
            GenerateVaspInput.set(self, **kwargs)
            self.results.clear()

    def reset(self):
        self.atoms = None
        self.clear_results()

    def clear_results(self):
        self.results.clear()
        self._xml_calc = None

    @contextmanager    #其中@启用，contextmanager对原来不是上下文管理器的类变成了一个上下文管理器
    def _txt_outstream(self):           #文本输出流
        """Custom function for opening a text output stream. Uses self.txt   #用于打开文本输出流的自定义函数。使用self.txt确定输出流，并接受字符串或打开的可写对象。
        to determine the output stream, and accepts a string or an open
        writable object.
        If a string is used, a new stream is opened, and automatically closes  #如果使用了字符串，则会打开一个新流，并在退出时再次自动关闭新流。
        the new stream again when exiting.

        Examples:
        # Pass a string                    传递一个字符串
        calc.txt = 'vasp.out'
        with calc.txt_outstream() as out:       将calc.txt_outstream()作为输出:
            calc.run(out=out)   # Redirects the stdout to 'vasp.out'   将标准输出重定向到'vasp.out'

        # Use an existing stream        使用现有的流
        mystream = open('vasp.out', 'w')
        calc.txt = mystream
        with calc.txt_outstream() as out:
            calc.run(out=out)
        mystream.close()

        # Print to stdout        打印到标准输出
        calc.txt = '-'
        with calc.txt_outstream() as out:
            calc.run(out=out)   # output is written to stdout    输出写入标准输出
        """

        txt = self.txt
        open_and_close = False  # Do we open the file?     打开文件了？

        if txt is None:
            # Suppress stdout       抑制标准输出
            out = subprocess.DEVNULL    #子流程
        else:
            if isinstance(txt, str):           #判断是否是实例
                if txt == '-':
                    # subprocess.call redirects this to stdout   子流程。Call将此重定向到标准输出
                    out = None
                else:
                    # Open the file in the work directory           在工作目录中打开该文件
                    txt = self._indir(txt)
                    # We wait with opening the file, until we are inside the           我们等待打开文件，直到我们进入try
                    # try/finally
                    open_and_close = True   #开关启用了？
            elif hasattr(txt, 'write'):    #函数 /检查对象是否拥有某个属性 /判断类实例中是否含有某个属性
                out = txt
            else:
                raise RuntimeError('txt should either be a string'              #运行时错误   'txt应该是一个字符串' #'或I/O流，得到{}
                                   'or an I/O stream, got {}'.format(txt))

        try:
            if open_and_close:
                out = open(txt, 'w')
            yield out
        finally:
            if open_and_close:
                out.close()

-------------------------------------------------------------------------------------------------------------------
    def calculate(self,
                  atoms=None,
                  properties=('energy', ),
                  system_changes=tuple(calculator.all_changes)):
        """Do a VASP calculation in the specified directory.           在指定目录下进行VASP计算。

        This will generate the necessary VASP input files, and then               这将生成必要的VASP输入文件
        execute VASP. After execution, the energy, forces. etc. are read          然后执行 VASP。执行后，能量，力等从 VASP 输出文件中读取。
        from the VASP output files.
        """
        # Check for zero-length lattice vectors and PBC   检查零长度晶格向量和PBC
        # and that we actually have an Atoms object.      以及我们实际上有一个 Atoms 对象。
        check_atoms(atoms)

        self.clear_results()

        if atoms is not None:
            self.atoms = atoms.copy()

        command = self.make_command(self.command)
        self.write_input(self.atoms, properties, system_changes)      #写入input？

        with self._txt_outstream() as out:                              #将self._txt_outstream()作为输出:
            errorcode = self._run(command=command,                      #错误代码
                                  out=out,
                                  directory=self.directory)

        if errorcode:
            raise calculator.CalculationFailed(                          #计算失败
                '{} in {} returned an error: {:d}'.format(                  #返回错误:{:d}
                    self.name, self.directory, errorcode))

        # Read results from calculation            读取计算结果
        self.update_atoms(atoms)
        self.read_results()

---------------------------------------------------------------------------------------------------------
    def _run(self, command=None, out=None, directory=None):
        """Method to explicitly execute VASP"""         #显示执行vasp的方法
        if command is None:
            command = self.command
        if directory is None:
            directory = self.directory
        errorcode = subprocess.call(command,        #子流程
                                    shell=True,
                                    stdout=out,
                                    cwd=directory)
        return errorcode

    def check_state(self, atoms, tol=1e-15):            #检查状态
        """Check for system changes since last calculation."""        #检查自上次以来对系统的更改
        def compare_dict(d1, d2):                                 #对比字典d1、d2  比较字典的辅助函数""
            """Helper function to compare dictionaries"""
            # Use symmetric difference to find keys which aren't shared   使用对称差分查找不共享的密钥
            # for python 2.7 compatibility      对于python 2.7兼容性
            if set(d1.keys()) ^ set(d2.keys()):
                return False

            # Check for differences in values        检查值的差异
            for key, value in d1.items():
                if np.any(value != d2[key]):
                    return False
            return True

        # First we check for default changes  首先，我们检查默认更改
        system_changes = Calculator.check_state(self, atoms, tol=tol)     #计算器检查状态

        # We now check if we have made any changes to the input parameters   现在我们检查是否对输入参数进行了任何更改
        # XXX: Should we add these parameters to all_changes?                我们是否应该将这些参数添加到all_changes?
        for param_string, old_dict in self.param_state.items():     #对于字符参数，旧字典
            param_dict = getattr(self, param_string)  # Get current param dict     获取当前参数目录     get attr属性
            if not compare_dict(param_dict, old_dict):      #如果没，就添加string
                system_changes.append(param_string)

        return system_changes

    def _store_param_state(self):                   #存储参数状态
        """Store current parameter state"""        #存储当前参数状态
        self.param_state = dict(
            float_params=self.float_params.copy(),       #复制相关参数，赋值
            exp_params=self.exp_params.copy(),
            string_params=self.string_params.copy(),
            int_params=self.int_params.copy(),
            input_params=self.input_params.copy(),
            bool_params=self.bool_params.copy(),
            list_int_params=self.list_int_params.copy(),
            list_bool_params=self.list_bool_params.copy(),
            list_float_params=self.list_float_params.copy(),
            dict_params=self.dict_params.copy())

    def asdict(self):                                   #as dict 作为字典/目录
        """Return a dictionary representation of the calculator state.     返回计算器状态的字典表示
        Does NOT contain information on the ``command``, ``txt`` or         不包含。。。。。的关键词
        ``directory`` keywords.
        Contains the following keys:                    包含下列关键词

            - ``ase_version``                           ase版本、vasp版本 输入
            - ``vasp_version``
            - ``inputs``
            - ``results``
            - ``atoms`` (Only if the calculator has an ``Atoms`` object)    只有当计算器有一个' '原子' '对象
        """
        # Get versions         获取版本
        asevers = ase.__version__
        vaspvers = self.get_version()

        self._store_param_state()  # Update param state     更新参数状态
        # Store input parameters which have been set       存储已设置的输入参数
        inputs = {
            key: value
            for param_dct in self.param_state.values()
            for key, value in param_dct.items() if value is not None
        }

        dct = {
            'ase_version': asevers,
            'vasp_version': vaspvers,
            # '__ase_objtype__': self.ase_objtype,           ase项目类型
            'inputs': inputs,
            'results': self.results.copy()
        }

        if self.atoms:
            # Encode atoms as dict             将原子编码为目录
            from ase.db.row import atoms2dict           #原子 to 字典
            dct['atoms'] = atoms2dict(self.atoms)

        return dct

    def fromdict(self, dct):
        """Restore calculator from a :func:`~ase.calculators.vasp.Vasp.asdict`    从。。。目录恢复计算器
        dictionary.

        Parameters:

        dct: Dictionary
            The dictionary which is used to restore the calculator state.     用于恢复计算器状态的字典。
        """
        if 'vasp_version' in dct:
            self.version = dct['vasp_version']
        if 'inputs' in dct:
            self.set(**dct['inputs'])
            self._store_param_state()
        if 'atoms' in dct:
            from ase.db.row import AtomsRow
            atoms = AtomsRow(dct['atoms']).toatoms()
            self.atoms = atoms
        if 'results' in dct:
            self.results.update(dct['results'])

    def write_json(self, filename):
        """Dump calculator state to JSON file.        将计算器状态转储到JSON文件

        Parameters:

        filename: string
            The filename which the JSON file will be stored to.  JSON文件将存储到的文件名。
            Prepends the ``directory`` path to the filename.     将' ' directory ' '路径添加到文件名。
        """
        filename = self._indir(filename)
        dct = self.asdict()
        jsonio.write_json(filename, dct)

    def read_json(self, filename):
        """Load Calculator state from an exported JSON Vasp file."""   #从导出的JSON Vasp文件中加载计算器状态。
        dct = jsonio.read_json(filename)
        self.fromdict(dct)

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write VASP inputfiles, INCAR, KPOINTS and POTCAR"""       #写VASP的输入文件, INCAR, KPOINTS和POTCAR
        # Create the folders where we write the files, if we aren't in the   如果不在当前工作目录中，创建用于写入文件的文件夹。
        # current working directory.
        if self.directory != os.curdir and not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        self.initialize(atoms)

        GenerateVaspInput.write_input(self, atoms, directory=self.directory)    #vasp输入文件生成器

    def read(self, label=None):
        """Read results from VASP output files.   从VASP输出文件读取结果。
        Files which are read: OUTCAR, CONTCAR and vasprun.xml   读取的文件:OUTCAR、CONTCAR和vasprin .xml
        Raises ReadError if they are not found"""     #如果没有找到它们，将引发读取错误
        if label is None:
            label = self.label
        Calculator.read(self, label)

        # If we restart, self.parameters isn't initialized   如果我们重新开始，self.参数不初始化
        if self.parameters is None:
            self.parameters = self.get_default_parameters()           #没参数自动获取默认值

        # Check for existence of the necessary output files        检查是否存在必要的输出文件
        for f in ['OUTCAR', 'CONTCAR', 'vasprun.xml']:
            file = self._indir(f)
            if not file.is_file():
                raise calculator.ReadError(
                    'VASP outputfile {} was not found'.format(file))     #没找到，将引发读取错误  VASP 输出文件（）未找到     

        # Build sorting and resorting lists     建立排序和求助/依赖列表
        self.read_sort()

        # Read atoms
        self.atoms = self.read_atoms(filename=self._indir('CONTCAR'))       #读原子在contcar里读取？

        # Read parameters
        self.read_incar(filename=self._indir('INCAR'))       #in dir 目录下的incar
        self.read_kpoints(filename=self._indir('KPOINTS'))
        self.read_potcar(filename=self._indir('POTCAR'))

        # Read the results from the calculation     读取计算结果
        self.read_results()                

    def _indir(self, filename):
        """Prepend current directory to filename"""             #将当前目录添加到文件名""
        return Path(self.directory) / filename

    def read_sort(self):
        """Create the sorting and resorting list from ase-sort.dat.         从ase-sort.dat创建排序和求助列表
        If the ase-sort.dat file does not exist, the sorting is redone.     如果ase-sort.dat文件不存在，则重新进行排序
        """
        sortfile = self._indir('ase-sort.dat')    #排序文件
        if os.path.isfile(sortfile):                #判断指定的文件是否存在且为文件
            self.sort = []
            self.resort = []
            with open(sortfile, 'r') as fd:
                for line in fd:
                    sort, resort = line.split()
                    self.sort.append(int(sort))
                    self.resort.append(int(resort))
        else:
            # Redo the sorting          #重新排序
            atoms = read(self._indir('CONTCAR'))
            self.initialize(atoms)

    def read_atoms(self, filename):
        """Read the atoms from file located in the VASP    从位于VASP工作目录中的文件中读取原子。通常称为CONTCAR
        working directory. Normally called CONTCAR."""
        return read(filename)[self.resort]

    def update_atoms(self, atoms):
        """Update the atoms object with new positions and cell"""     #用新的位置和单元格更新原子对象
        if (self.int_params['ibrion'] is not None
                and self.int_params['nsw'] is not None):
            if self.int_params['ibrion'] > -1 and self.int_params['nsw'] > 0:
                # Update atomic positions and unit cell with the ones read   用从contcar中读取的数据更新原子位置和单元格
                # from CONTCAR.
                atoms_sorted = read(self._indir('CONTCAR'))
                atoms.positions = atoms_sorted[self.resort].positions
                atoms.cell = atoms_sorted.cell

        self.atoms = atoms  # Creates a copy

    def read_results(self):
        """Read the results from VASP output files"""   #从VASP输出文件中读取结果
        # Temporarily load OUTCAR into memory          #暂时将OUTCAR加载到内存中
        outcar = self.load_file('OUTCAR')

        # Read the data we can from vasprun.xml   读取vasprune .xml中的数据
        calc_xml = self._read_xml()
        xml_results = calc_xml.results

        # Fix sorting           修复排序
        xml_results['forces'] = xml_results['forces'][self.resort]

        self.results.update(xml_results)

        # Parse the outcar, as some properties are not loaded in vasprun.xml 解析outcar，因为vasprin .xml中没有加载一些属性
        # We want to limit this as much as possible, as reading large OUTCAR's
        # is relatively slow
        # Removed for now
        # self.read_outcar(lines=outcar)                我们希望尽可能地限制这一点，因为读取较大的OUTCAR相对较慢

        # Update results dict with results from OUTCAR   使用来自OUTCAR的结果更新结果字典
        # which aren't written to the atoms object we read from   这些结果未写入我们从 vasprun.xml 文件中读取的原子对象。
        # the vasprun.xml file.

        self.converged = self.read_convergence(lines=outcar)
        self.version = self.read_version()
        magmom, magmoms = self.read_mag(lines=outcar)    #磁
        dipole = self.read_dipole(lines=outcar)         #dipole双极子
        nbands = self.read_nbands(lines=outcar)          #nband，算光学有用
        self.results.update(
            dict(magmom=magmom, magmoms=magmoms, dipole=dipole, nbands=nbands))

        # Stress is not always present.   压力不总是存在
        # Prevent calculation from going into a loop  防止计算进入循环
        if 'stress' not in self.results:
            self.results.update(dict(stress=None))

        self._set_old_keywords()

        # Store the parameters used for this calculation  存储用于此计算的参数
        self._store_param_state()

    def _set_old_keywords(self):
        """Store keywords for backwards compatibility wd VASP calculator"""  #存储关键字的向后兼容性wd VASP计算器
        self.spinpol = self.get_spin_polarized()                                #spin_polarized自旋极化
        self.energy_free = self.get_potential_energy(force_consistent=True)     #自由能
        self.energy_zero = self.get_potential_energy(force_consistent=False)    #零能点？
        self.forces = self.get_forces()     #力
        self.fermi = self.get_fermi_level()                 #费米
        self.dipole = self.get_dipole_moment()              #双极子
        # Prevent calculation from going into a loop
        self.stress = self.get_property('stress', allow_calculation=False)
        self.nbands = self.get_number_of_bands()      #能带的个数？序数？

    # Below defines some functions for faster access to certain common keywords 下面定义了一些用于更快地访问某些常见关键字的函数
    @property     #python的@property是python的一种装饰器，是用来修饰方法的。
    def kpts(self):
        """Access the kpts from input_params dict"""   #从输入参数字典访问kpts K点
        return self.input_params['kpts']

    @kpts.setter      #setter 设值函数
    def kpts(self, kpts):
        """Set kpts in input_params dict"""     #设置input_params dict中的kpts
        self.input_params['kpts'] = kpts

    @property
    def encut(self):
        """Direct access to the encut parameter"""    #直接访问encut参数，这个截断能参数从哪冒出来的？
        return self.float_params['encut']

    @encut.setter
    def encut(self, encut):
        """Direct access for setting the encut parameter"""   #直接访问设置encut参数
        self.set(encut=encut)

    @property
    def xc(self):
        """Direct access to the xc parameter"""   #直接访问xc参数  对应xc='PBE'
        return self.get_xc_functional()

    @xc.setter
    def xc(self, xc):
        """Direct access for setting the xc parameter"""   #直接访问xc参数的设置
        self.set(xc=xc)

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        if atoms is None:
            self._atoms = None
            self.clear_results()       #清除结果？
        else:
            if self.check_state(atoms):
                self.clear_results()
            self._atoms = atoms.copy()  #把原子从某个地方复制过来？

    def load_file(self, filename):
        """Reads a file in the directory, and returns the lines   读取目录中的文件，并返回行数

        Example:
        >>> outcar = load_file('OUTCAR')
        """
        filename = self._indir(filename)
        with open(filename, 'r') as fd:
            return fd.readlines()          #返回行数

    @contextmanager
    def load_file_iter(self, filename):
        """Return a file iterator"""            #返回一个文件迭代器"  文件迭代器又是啥？

        filename = self._indir(filename)
        with open(filename, 'r') as fd:
            yield fd

    def read_outcar(self, lines=None):
        """Read results from the OUTCAR file.   从OUTCAR文件读取结果。
        Deprecated, see read_results()"""         #弃用，类似read_results????
        if not lines:
            lines = self.load_file('OUTCAR')
        # Spin polarized calculation?     自旋极化计算
        self.spinpol = self.get_spin_polarized()

        self.version = self.get_version()

        # XXX: Do we want to read all of this again?    我们要把这些再读一遍吗?
        self.energy_free, self.energy_zero = self.read_energy(lines=lines)
        self.forces = self.read_forces(lines=lines)
        self.fermi = self.read_fermi(lines=lines)               #lines = lines 行值赋行值？

        self.dipole = self.read_dipole(lines=lines)

        self.stress = self.read_stress(lines=lines)
        self.nbands = self.read_nbands(lines=lines)

        self.read_ldau()
        self.magnetic_moment, self.magnetic_moments = self.read_mag(
            lines=lines)

    def _read_xml(self) -> SinglePointDFTCalculator:       #vasprun.xml?   单点DFT计算
        """Read vasprun.xml, and return the last calculator object.   读取vasprin .xml，并返回最后一个计算器对象
        Returns calculator from the xml file.           从xml文件返回计算器。
        Raises a ReadError if the reader is not able to construct a calculator.  如果读取器无法构造计算器，则引发ReadError
        """
        file = self._indir('vasprun.xml')
        incomplete_msg = (              #不完全信息？
            f'The file "{file}" is incomplete, and no DFT data was available. '         #f'文件“{file}”不完整，没有DFT数据可用。
            'This is likely due to an incomplete calculation.')             #这可能是由于不完整的计算
        try:
            _xml_atoms = read(file, index=-1, format='vasp-xml')
            # Silence mypy, we should only ever get a single atoms object   我们应该只得到单个原子的物体
            assert isinstance(_xml_atoms, ase.Atoms)
        except ElementTree.ParseError as exc:                   #元素树，语法分析错误？
            raise calculator.ReadError(incomplete_msg) from exc

        if _xml_atoms is None or _xml_atoms.calc is None:
            raise calculator.ReadError(incomplete_msg)

        self._xml_calc = _xml_atoms.calc
        return self._xml_calc

    @property
    def _xml_calc(self) -> SinglePointDFTCalculator:
        if self.__xml_calc is None:
            raise RuntimeError(('vasprun.xml data has not yet been loaded. '   #xml数据还没有加载
                                'Run read_results() first.'))               #首先运行read_results ()
        return self.__xml_calc

    @_xml_calc.setter
    def _xml_calc(self, value):     #xml里面的calc？
        self.__xml_calc = value

    def get_ibz_k_points(self):         #ibz不知是啥，只认识k点
        calc = self._xml_calc
        return calc.get_ibz_k_points()

    def get_kpt(self, kpt=0, spin=0):           #自旋=0
        calc = self._xml_calc
        return calc.get_kpt(kpt=kpt, spin=spin)

    def get_eigenvalues(self, kpt=0, spin=0):   #eigenvalue  特征值
        calc = self._xml_calc
        return calc.get_eigenvalues(kpt=kpt, spin=spin)

    def get_fermi_level(self):
        calc = self._xml_calc
        return calc.get_fermi_level()

    def get_homo_lumo(self):        #homo_lumo 最高占据分子轨道_最低占据分子轨道
        calc = self._xml_calc
        return calc.get_homo_lumo()

    def get_homo_lumo_by_spin(self, spin=0):   #通过自旋的最高占据分子轨道_最低占据分子轨道
        calc = self._xml_calc
        return calc.get_homo_lumo_by_spin(spin=spin)

    def get_occupation_numbers(self, kpt=0, spin=0):        #占据数
        calc = self._xml_calc
        return calc.get_occupation_numbers(kpt, spin)

    def get_spin_polarized(self):               #自旋极化
        calc = self._xml_calc
        return calc.get_spin_polarized()

    def get_number_of_spins(self):              #自旋数
        calc = self._xml_calc
        return calc.get_number_of_spins()

    def get_number_of_bands(self):      #能带数
        return self.results.get('nbands', None)

    def get_number_of_electrons(self, lines=None):         #电子数 
        if not lines:
            lines = self.load_file('OUTCAR')

        nelect = None
        for line in lines:          #后面加了s的意思好像是？
            if 'total number of electrons' in line:     #价电子总数
                nelect = float(line.split('=')[1].split()[0].strip())
                break
        return nelect

    def get_k_point_weights(self):          #k点权重？
        filename = self._indir('IBZKPT')            #IBZKPT文件
        return self.read_k_point_weights(filename)

    def get_dos(self, spin=None, **kwargs):
        """
        The total DOS.              #DOS是啥？磁盘操作系统？

        Uses the ASE DOS module, and returns a tuple with  使用ASE DOS模块，并返回一个元组with (能量、dos)
        (energies, dos).
        """
        from ase.dft.dos import DOS
        dos = DOS(self, **kwargs)
        e = dos.get_energies()
        d = dos.get_dos(spin=spin)
        return e, d

    def get_version(self):
        if self.version is None:
            # Try if we can read the version number
            self.version = self.read_version()
        return self.version

    def read_version(self):
        """Get the VASP version number"""
        # The version number is the first occurrence, so we can just
        # load the OUTCAR, as we will return soon anyway
        if not os.path.isfile(self._indir('OUTCAR')):
            return None
        with self.load_file_iter('OUTCAR') as lines:
            for line in lines:
                if ' vasp.' in line:
                    return line[len(' vasp.'):].split()[0]
        # We didn't find the version in VASP
        return None

    def get_number_of_iterations(self):
        return self.read_number_of_iterations()

    def read_number_of_iterations(self):
        niter = None
        with self.load_file_iter('OUTCAR') as lines:
            for line in lines:
                # find the last iteration number
                if '- Iteration' in line:
                    niter = list(map(int, re.findall(r'\d+', line)))[1]
        return niter

    def read_number_of_ionic_steps(self):
        niter = None
        with self.load_file_iter('OUTCAR') as lines:
            for line in lines:
                if '- Iteration' in line:
                    niter = list(map(int, re.findall(r'\d+', line)))[0]
        return niter

    def read_stress(self, lines=None):
        """Read stress from OUTCAR.

        Depreciated: Use get_stress() instead.
        """
        # We don't really need this, as we read this from vasprun.xml
        # keeping it around "just in case" for now
        if not lines:
            lines = self.load_file('OUTCAR')

        stress = None
        for line in lines:
            if ' in kB  ' in line:
                stress = -np.array([float(a) for a in line.split()[2:]])
                stress = stress[[0, 1, 2, 4, 5, 3]] * 1e-1 * ase.units.GPa
        return stress

    def read_ldau(self, lines=None):
        """Read the LDA+U values from OUTCAR"""
        if not lines:
            lines = self.load_file('OUTCAR')

        ldau_luj = None
        ldauprint = None
        ldau = None
        ldautype = None
        atomtypes = []
        # read ldau parameters from outcar
        for line in lines:
            if line.find('TITEL') != -1:  # What atoms are present
                atomtypes.append(line.split()[3].split('_')[0].split('.')[0])
            if line.find('LDAUTYPE') != -1:  # Is this a DFT+U calculation
                ldautype = int(line.split('=')[-1])
                ldau = True
                ldau_luj = {}
            if line.find('LDAUL') != -1:
                L = line.split('=')[-1].split()
            if line.find('LDAUU') != -1:
                U = line.split('=')[-1].split()
            if line.find('LDAUJ') != -1:
                J = line.split('=')[-1].split()
        # create dictionary
        if ldau:
            for i, symbol in enumerate(atomtypes):
                ldau_luj[symbol] = {
                    'L': int(L[i]),
                    'U': float(U[i]),
                    'J': float(J[i])
                }
            self.dict_params['ldau_luj'] = ldau_luj

        self.ldau = ldau
        self.ldauprint = ldauprint
        self.ldautype = ldautype
        self.ldau_luj = ldau_luj
        return ldau, ldauprint, ldautype, ldau_luj

    def get_xc_functional(self):
        """Returns the XC functional or the pseudopotential type

        If a XC recipe is set explicitly with 'xc', this is returned.
        Otherwise, the XC functional associated with the
        pseudopotentials (LDA, PW91 or PBE) is returned.
        The string is always cast to uppercase for consistency
        in checks."""
        if self.input_params.get('xc', None):
            return self.input_params['xc'].upper()
        if self.input_params.get('pp', None):
            return self.input_params['pp'].upper()
        raise ValueError('No xc or pp found.')

    # Methods for reading information from OUTCAR files:
    def read_energy(self, all=None, lines=None):
        """Method to read energy from OUTCAR file.
        Depreciated: use get_potential_energy() instead"""
        if not lines:
            lines = self.load_file('OUTCAR')

        [energy_free, energy_zero] = [0, 0]
        if all:
            energy_free = []
            energy_zero = []
        for line in lines:
            # Free energy
            if line.lower().startswith('  free  energy   toten'):
                if all:
                    energy_free.append(float(line.split()[-2]))
                else:
                    energy_free = float(line.split()[-2])
            # Extrapolated zero point energy
            if line.startswith('  energy  without entropy'):
                if all:
                    energy_zero.append(float(line.split()[-1]))
                else:
                    energy_zero = float(line.split()[-1])
        return [energy_free, energy_zero]

    def read_forces(self, all=False, lines=None):
        """Method that reads forces from OUTCAR file.

        If 'all' is switched on, the forces for all ionic steps
        in the OUTCAR file be returned, in other case only the
        forces for the last ionic configuration is returned."""

        if not lines:
            lines = self.load_file('OUTCAR')

        if all:
            all_forces = []

        for n, line in enumerate(lines):
            if 'TOTAL-FORCE' in line:
                forces = []
                for i in range(len(self.atoms)):
                    forces.append(
                        np.array(
                            [float(f) for f in lines[n + 2 + i].split()[3:6]]))

                if all:
                    all_forces.append(np.array(forces)[self.resort])

        if all:
            return np.array(all_forces)
        return np.array(forces)[self.resort]

    def read_fermi(self, lines=None):
        """Method that reads Fermi energy from OUTCAR file"""
        if not lines:
            lines = self.load_file('OUTCAR')

        E_f = None
        for line in lines:
            if 'E-fermi' in line:
                E_f = float(line.split()[2])
        return E_f

    def read_dipole(self, lines=None):
        """Read dipole from OUTCAR"""
        if not lines:
            lines = self.load_file('OUTCAR')

        dipolemoment = np.zeros([1, 3])
        for line in lines:
            if 'dipolmoment' in line:
                dipolemoment = np.array([float(f) for f in line.split()[1:4]])
        return dipolemoment

    def read_mag(self, lines=None):
        if not lines:
            lines = self.load_file('OUTCAR')
        p = self.int_params
        q = self.list_float_params
        if self.spinpol:
            magnetic_moment = self._read_magnetic_moment(lines=lines)
            if ((p['lorbit'] is not None and p['lorbit'] >= 10)
                    or (p['lorbit'] is None and q['rwigs'])):
                magnetic_moments = self._read_magnetic_moments(lines=lines)
            else:
                warn(('Magnetic moment data not written in OUTCAR (LORBIT<10),'
                      ' setting magnetic_moments to zero.\nSet LORBIT>=10'
                      ' to get information on magnetic moments'))
                magnetic_moments = np.zeros(len(self.atoms))
        else:
            magnetic_moment = 0.0
            magnetic_moments = np.zeros(len(self.atoms))
        return magnetic_moment, magnetic_moments

    def _read_magnetic_moments(self, lines=None):
        """Read magnetic moments from OUTCAR.
        Only reads the last occurrence. """
        if not lines:
            lines = self.load_file('OUTCAR')

        magnetic_moments = np.zeros(len(self.atoms))
        magstr = 'magnetization (x)'

        # Search for the last occurrence
        nidx = -1
        for n, line in enumerate(lines):
            if magstr in line:
                nidx = n

        # Read that occurrence
        if nidx > -1:
            for m in range(len(self.atoms)):
                magnetic_moments[m] = float(lines[nidx + m + 4].split()[4])
        return magnetic_moments[self.resort]

    def _read_magnetic_moment(self, lines=None):
        """Read magnetic moment from OUTCAR"""
        if not lines:
            lines = self.load_file('OUTCAR')

        for n, line in enumerate(lines):
            if 'number of electron  ' in line:
                magnetic_moment = float(line.split()[-1])
        return magnetic_moment

    def read_nbands(self, lines=None):
        """Read number of bands from OUTCAR"""
        if not lines:
            lines = self.load_file('OUTCAR')

        for line in lines:
            line = self.strip_warnings(line)
            if 'NBANDS' in line:
                return int(line.split()[-1])
        return None

    def read_convergence(self, lines=None):
        """Method that checks whether a calculation has converged."""
        if not lines:
            lines = self.load_file('OUTCAR')

        converged = None
        # First check electronic convergence
        for line in lines:
            if 0:  # vasp always prints that!
                if line.rfind('aborting loop') > -1:  # scf failed
                    raise RuntimeError(line.strip())
                    break
            if 'EDIFF  ' in line:
                ediff = float(line.split()[2])
            if 'total energy-change' in line:
                # I saw this in an atomic oxygen calculation. it
                # breaks this code, so I am checking for it here.
                if 'MIXING' in line:
                    continue
                split = line.split(':')
                a = float(split[1].split('(')[0])
                b = split[1].split('(')[1][0:-2]
                # sometimes this line looks like (second number wrong format!):
                # energy-change (2. order) :-0.2141803E-08  ( 0.2737684-111)
                # we are checking still the first number so
                # let's "fix" the format for the second one
                if 'e' not in b.lower():
                    # replace last occurrence of - (assumed exponent) with -e
                    bsplit = b.split('-')
                    bsplit[-1] = 'e' + bsplit[-1]
                    b = '-'.join(bsplit).replace('-e', 'e-')
                b = float(b)
                if [abs(a), abs(b)] < [ediff, ediff]:
                    converged = True
                else:
                    converged = False
                    continue
        # Then if ibrion in [1,2,3] check whether ionic relaxation
        # condition been fulfilled
        if ((self.int_params['ibrion'] in [1, 2, 3]
             and self.int_params['nsw'] not in [0])):
            if not self.read_relaxed():
                converged = False
            else:
                converged = True
        return converged

    def read_k_point_weights(self, filename):
        """Read k-point weighting. Normally named IBZKPT."""

        lines = self.load_file(filename)

        if 'Tetrahedra\n' in lines:
            N = lines.index('Tetrahedra\n')
        else:
            N = len(lines)
        kpt_weights = []
        for n in range(3, N):
            kpt_weights.append(float(lines[n].split()[3]))
        kpt_weights = np.array(kpt_weights)
        kpt_weights /= np.sum(kpt_weights)

        return kpt_weights

    def read_relaxed(self, lines=None):
        """Check if ionic relaxation completed"""
        if not lines:
            lines = self.load_file('OUTCAR')
        for line in lines:
            if 'reached required accuracy' in line:
                return True
        return False

    def read_spinpol(self, lines=None):
        """Method which reads if a calculation from spinpolarized using OUTCAR.

        Depreciated: Use get_spin_polarized() instead.
        """
        if not lines:
            lines = self.load_file('OUTCAR')

        for line in lines:
            if 'ISPIN' in line:
                if int(line.split()[2]) == 2:
                    self.spinpol = True
                else:
                    self.spinpol = False
        return self.spinpol

    def strip_warnings(self, line):
        """Returns empty string instead of line from warnings in OUTCAR."""
        if line[0] == "|":
            return ""
        return line

    @property
    def txt(self):
        return self._txt

    @txt.setter
    def txt(self, txt):
        if isinstance(txt, PurePath):
            txt = str(txt)
        self._txt = txt

    def get_number_of_grid_points(self):
        raise NotImplementedError

    def get_pseudo_density(self):
        raise NotImplementedError

    def get_pseudo_wavefunction(self, n=0, k=0, s=0, pad=True):
        raise NotImplementedError

    def get_bz_k_points(self):
        raise NotImplementedError

    def read_vib_freq(self, lines=None):
        """Read vibrational frequencies.

        Returns list of real and list of imaginary frequencies."""
        freq = []
        i_freq = []

        if not lines:
            lines = self.load_file('OUTCAR')

        for line in lines:
            data = line.split()
            if 'THz' in data:
                if 'f/i=' not in data:
                    freq.append(float(data[-2]))
                else:
                    i_freq.append(float(data[-2]))
        return freq, i_freq

    def get_nonselfconsistent_energies(self, bee_type):
        """ Method that reads and returns BEE energy contributions
            written in OUTCAR file.
        """
        assert bee_type == 'beefvdw'
        cmd = 'grep -32 "BEEF xc energy contributions" OUTCAR | tail -32'
        p = os.popen(cmd, 'r')
        s = p.readlines()
        p.close()
        xc = np.array([])
        for line in s:
            l_ = float(line.split(":")[-1])
            xc = np.append(xc, l_)
        assert len(xc) == 32
        return xc


#####################################
# Below defines helper functions
# for the VASP calculator
#####################################


def check_atoms(atoms: ase.Atoms) -> None:
    """Perform checks on the atoms object, to verify that
    it can be run by VASP.
    A CalculatorSetupError error is raised if the atoms are not supported.
    """

    # Loop through all check functions
    for check in (check_atoms_type, check_cell, check_pbc):
        check(atoms)


def check_cell(atoms: ase.Atoms) -> None:
    """Check if there is a zero unit cell.
    Raises CalculatorSetupError if the cell is wrong.
    """
    if atoms.cell.rank < 3:
        raise calculator.CalculatorSetupError(
            "The lattice vectors are zero! "
            "This is the default value - please specify a "
            "unit cell.")


def check_pbc(atoms: ase.Atoms) -> None:
    """Check if any boundaries are not PBC, as VASP
    cannot handle non-PBC.
    Raises CalculatorSetupError.
    """
    if not atoms.pbc.all():
        raise calculator.CalculatorSetupError(
            "Vasp cannot handle non-periodic boundaries. "
            "Please enable all PBC, e.g. atoms.pbc=True")


def check_atoms_type(atoms: ase.Atoms) -> None:
    """Check that the passed atoms object is in fact an Atoms object.
    Raises CalculatorSetupError.
    """
    if not isinstance(atoms, ase.Atoms):
        raise calculator.CalculatorSetupError(
            ('Expected an Atoms object, '
             'instead got object of type {}'.format(type(atoms))))
