# A simple python driver for the Quonset Microwave Cantenna

Radar documentation [here](https://www.quonsetmicrowave.com/QM-RDKIT-p/qm-rdkit.htm).

---

### To install conda env:

```bash
conda env create -f cantenna.yml
```

---

### Usage

1. Activate conda env that was just installed:

    ```bash
    conda activate cantenna
    ```

2. Use `list.py` to print a list of detected VISA devices:

    ```bash
    python list.py
    ```

    ```
    ('ASRL/dev/cu.debug-console::INSTR', 'ASRL/dev/cu.Bluetooth-Incoming-Port::INSTR', 'USB0::8210::19::0087::0::INSTR')
    ```

    In this case, the last device is the radar, with 0087 being the serial number.

3. Collect and plot data:

    A basic live range-doppler plot can be generated via the following:

    ```bash
    python collect.py -a USB0::8210::19::0087::0::INSTR
    ```

    The above device addres is the default (the one we had in the project) if `collect.py` is called without arguments.

4. Advanced usage:

    Other arguments for more advanced usage can be accessed via the following:

    ```bash
    python collect.py -h
    ```