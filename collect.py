import time

import pyvisa

import numpy as np
from math import ceil
import scipy.signal.windows as wn

import argparse
from pathlib import Path

def get_volatage(
    radar: pyvisa.resources.usb.USBInstrument,
    start_f_GHz: float = 2.4, stop_f_GHz: float = 2.5,
    mode: str = "RAMP", ramp_time_ms: float = 16., num_sweeps: int = 1
):
    samples_per_read = 31
    sample_time = 50e-6
    sample_freq = 1/sample_time

    if start_f_GHz < 2.25 or stop_f_GHz > 2.5 or start_f_GHz >= stop_f_GHz:
        raise ValueError(
            "start freq and stop freq must be between"
            " 2.4 and 2.5 with start < stop."
        )

    if mode == "RAMP":
        mode = 0
    elif mode == "TRI":
        mode = 1
    elif mode == "AUTOTRI":
        mode = 2
    else:
        raise ValueError('Mode must be "RAMP", "TRI", or "AUTOTRI".')

    max_ramp_samples = 4096
    ramp_samples = ceil((ramp_time_ms/1000)*sample_freq)
    if ramp_samples > max_ramp_samples:
        raise ValueError(
            "Max allowable (ramp_time_s*sample_freq_Hz)"
            f" is {max_ramp_samples}."
        )
    
    radar.timeout = 10000
    radar.read_termination = '\n'
    radar.write_termination = '\n'

    if mode == 2:
        radar.write(f'SWEEP:FREQSTAR {start_f_GHz}')
        radar.write(f'SWEEP:FREQSTOP {stop_f_GHz}')
        radar.write(f'SWEEP:RAMPTIME {ramp_time_ms}')
        radar.write(f'SWEEP:TYPE {mode}')

        num_chirps = max_ramp_samples // ramp_samples
        samples_to_read = num_chirps*ramp_samples
        num_reads = ceil(samples_to_read / 31)

        start_time = time.perf_counter_ns()
        hex_strings = [[]]*num_sweeps
        big_time_ns = [0.]*num_sweeps
        hex_string = [""]*num_reads

        #execute data collection
        for n in range(num_sweeps):
            big_time_ns[n] = time.perf_counter_ns() - start_time
            radar.write('SWEEP:START')
            radar.write(f'CAPT:FRAM {samples_to_read}')
            for r in range(num_reads):
                radar.write(f'CAPT:FRAM?')
                hex_string[r] = radar.read()
            radar.write('SWEEP:STOP')
            hex_strings[n] = hex_string.copy()

        #format data
        for n, hex_string in enumerate(hex_strings):
            concat = ""
            for chunk in hex_string:
                concat += chunk.strip()
            chars = np.array(list(concat[0:samples_to_read*4]), dtype='U1').reshape(samples_to_read,4)
            hex_strings[n] = chars.view('U4').ravel()
        hex_strings = np.array(hex_strings)
        values = np.vectorize(lambda x: int(x, 16))(hex_strings)
        voltages = values/65535*5-2.5

        voltages = voltages.reshape(num_sweeps, num_chirps, ramp_samples)
        t1 = np.arange(ramp_samples)/sample_freq
        t2 = np.arange(num_chirps)/sample_freq*ramp_samples
        times = (t1, t2, np.array(big_time_ns)/1e9)


    else:
        radar.write(f'SWEEP:FREQSTAR {start_f_GHz}')
        radar.write(f'SWEEP:FREQSTOP {stop_f_GHz}')
        radar.write(f'SWEEP:RAMPTIME {ramp_time_ms}')
        radar.write(f'SWEEP:TYPE {mode}')

        num_reads = ceil(ramp_samples / samples_per_read)
        start_time = time.perf_counter_ns()
        hex_strings = [[]]*num_sweeps
        big_time_ns = [0.]*num_sweeps
        hex_string = [""]*num_reads

        #execute data collection
        for n in range(num_sweeps):
            big_time_ns[n] = time.perf_counter_ns() - start_time
            radar.write('SWEEP:START')
            radar.write(f'CAPT:FRAM {ramp_samples}')
            for r in range(num_reads):
                radar.write(f'CAPT:FRAM?')
                hex_string[r] = radar.read()
            hex_strings[n] = hex_string.copy()

        #format data
        for n, hex_string in enumerate(hex_strings):
            concat = ""
            for chunk in hex_string:
                concat += chunk.strip()
            chars = np.array(list(concat[0:ramp_samples*4]), dtype='U1').reshape(ramp_samples,4)
            hex_strings[n] = chars.view('U4').ravel()
        hex_strings = np.array(hex_strings)
        values = np.vectorize(lambda x: int(x, 16))(hex_strings)
        voltages = values/65535*5-2.5
        t1 = np.arange(ramp_samples)/sample_freq
        times = (t1, np.array(big_time_ns)/1e9)

    #returns voltage arrays and m start times for each dim
    return voltages, times, sample_freq

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    plt.ioff()

    mode_int_to_str = ["RAMP", "TRI", "AUTOTRI"]
    mode_str_to_int = {k: v for v, k in enumerate(mode_int_to_str)}

    window_int_to_str = ["rectangular", "hanning", "hamming"]
    window_str_to_int = {k: v for v, k in enumerate(window_int_to_str)}

    parser = argparse.ArgumentParser(
        description="Parses plotting options for radar."
    )
    parser.add_argument(
        '-a',
        '--address',
        type=str,
        help="Radar Address. Defaults to 'USB0::8210::19::0087::0::INSTR'.",
        default='USB0::8210::19::0087::0::INSTR'
    )

    parser.add_argument(
        '--start',
        type=float,
        help="Start frequency in GHz. Defaults to 2.3. Minimum 2.25.",
        default=2.3
    )

    parser.add_argument(
        '--stop',
        type=float,
        help="Stop frequency in GHz. Defaults to 2.5. Maximum 2.5.",
        default=2.5
    )

    parser.add_argument(
        '-rt',
        '--ramptime',
        type=int,
        help="Ramptime in ms. Default is 4ms. "
        "The maximum possible ramptime must satisfy "
        "ceil((ramp_time_ms/1000)*sample_freq) <= 4096. "
        "Sample_freq is 20000.",
        default=4
    )

    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        choices=['RAMP', 'TRI', 'AUTOTRI'],
        help="Either 'RAMP', 'TRI', or 'AUTOTRI'. "
        "Defaults to 'AUTOTRI'.",
        default="AUTOTRI"
    )

    parser.add_argument(
        '-w',
        '--window',
        type=str,
        choices=['rectangular', 'hanning', 'hamming'],
        help="Window to use for fft's. Defaults to rectangular",
        default='rectangular'
    )

    parser.add_argument(
        '-d',
        '--delay',
        type=float,
        help="Delay between sweeps in seconds. Defaults to 1.",
        default=1.
    )

    parser.add_argument(
        '-pt',
        '--plottype',
        type=str,
        choices = ['range', 'r', 'range-doppler', 'rd'],
        help="Either ('range' | 'r') or ('range-doppler'|'rd'). "
        "The latter requires -m to be 'AUTOTRI'. "
        "Default is 'rd'.",
        default='rd'
    )

    parser.add_argument(
        '-n',
        '--numsweeps',
        type=int,
        help="Number of sweeps to capture or plot."
        "Omitting will capture until plot window is closed.",
    )

    parser.add_argument(
        '-fp',
        '--figprefix',
        type=str,
        help="Figure file prefix (will append _n.png). "
        "If omitted, figures will not be saved.",
    )

    parser.add_argument(
        '-dp',
        '--dataprefix',
        type=str,
        help="Data file prefix (will append _n.npz). "
        "If omitted, data will not be saved. "
        "Data files can be opened with np.load()."
    )

    parser.add_argument(
        '-b',
        '--backgroundnoise',
        type=str,
        help="Provide a file path containing background returns or "
        "generate one if it doesn't exist. "
        "Generating a file will override --numsweeps and set it to 1. "
        "Providing an existing file will override --start, --stop, "
        "--ramptime, --mode, and --window. "
        "Supercedes any other passed config because "
        "saved array dimensions need to be equivalent."
    )

    def validate_and_resolve_path(path_str, needs_to_be_dir):
        if path_str is None:
            if needs_to_be_dir:
                return False, None
            else:
                return False, None, None

        path_obj = Path(path_str)

        if needs_to_be_dir:
            full_dir = path_obj.resolve()
            if not full_dir.exists():
                raise FileNotFoundError(f"The directory '{full_dir}' does not exist.")
            if not full_dir.is_dir():
                raise NotADirectoryError(f"The path '{full_dir}' exists but is not a directory.")
                
            return True, full_dir

        else:
            if path_str.endswith(('/', '\\')) or path_obj.is_dir():
                raise ValueError(
                    f"'{path_str}' appears to be a directory. "
                    "Please provide a file prefix."
                )
            full_path = path_obj.resolve()
            full_dir = full_path.parent
            if not full_dir.exists():
                raise FileNotFoundError(f"The parent directory '{full_dir}' does not exist.")
            if not full_dir.is_dir():
                raise NotADirectoryError(f"The parent path '{full_dir}' exists but is not a directory.")

            return True, full_dir, full_path

    args = parser.parse_args()

    #first things first check the background and override options
    create_bg = False
    bg_passed, _, bgpath = validate_and_resolve_path(args.backgroundnoise, False)
    if bg_passed:
        if bgpath.exists():
            print("Subtracting existing background measurement.")
            bgdata = np.load(bgpath)
            args.start = bgdata["start_f_GHz"].item()
            args.stop = bgdata["stop_f_GHz"].item()
            args.mode = mode_int_to_str[int(bgdata["mode"].item())]
            args.ramptime = bgdata["ramp_time_ms"].item()
            args.window = window_int_to_str[int(bgdata["window"].item())]
            bg_beats = bgdata["beats"]
        else:
            print(f"Creating new background measurement at {bgpath.name}.")
            create_bg = True
            args.numsweeps = 1

    start_f_GHz = args.start
    stop_f_GHz = args.stop
    mode = args.mode
    ramp_time_ms = args.ramptime

    if args.window == "rectangular":
        w = wn.boxcar
    elif args.window == "hanning":
        w = wn.hann
    elif args.window == "hamming":
        w = wn.hamming

    sweep_delay = args.delay
    if sweep_delay < 0.25:
        raise ValueError(
            "Sweep Delay must be at least 0.25 so data can be collected and processed."
        )

    if args.plottype == 'range' or args.plottype == 'r':
        plottype = 0
    else:
        plottype = 1
        if mode != "AUTOTRI":
            raise ValueError(
                "--mode must be 'AUTOTRI' for range-Doppler plots."
            )
    
    savefigs, _, fig_path = validate_and_resolve_path(args.figprefix, False)
    savedata, _, data_path = validate_and_resolve_path(args.dataprefix, False)

    if not (args.numsweeps is None):
        if args.numsweeps < 1:
            raise ValueError(f"Need minimum one sweep. You gave {args.numsweeps}.")

    rm = pyvisa.ResourceManager()
    try:
        radar = rm.open_resource(args.address)
    except Exception as e:
        raise ValueError(
            f"{args.address} could not be opened. "
            "Check the address or connect the radar."
        ) from e

    #data collection function
    def collect_data():
        voltage, _, sample_freq = get_volatage(
            radar, start_f_GHz, stop_f_GHz, mode, ramp_time_ms
        )

        voltage = voltage[0]
        if mode == 'AUTOTRI':
            voltage[1::2,:] = voltage[1::2,::-1]
        
        ramp_samples = voltage.shape[-1]
        nfft = ramp_samples*7
        beats = (2*np.fft.fft(w(ramp_samples)*voltage, nfft, axis=-1)/ramp_samples)[...,0:nfft//2]

        if bg_passed:
            if create_bg:
                print("Saving background measurement.")
                np.savez(
                    bgpath,
                    start_f_GHz=start_f_GHz,
                    stop_f_GHz=stop_f_GHz,
                    ramp_time_ms=ramp_time_ms,
                    mode=mode_str_to_int[mode],
                    window=window_str_to_int[args.window],
                    beats=beats
                )
            else:
                beats = beats-bg_beats

        if plottype == 0:
            power = 20*np.log10(np.abs(beats))
            if mode == 'AUTOTRI':
                power = np.mean(power, axis=0)
            return voltage, power, sample_freq, nfft
        elif plottype == 1:
            num_ramps = voltage.shape[0]
            nvfft = num_ramps
            range_doppler = 10*np.log10(
                np.abs(
                    np.fft.fftshift(
                        np.fft.fft(w(num_ramps).reshape(-1, 1)*beats, nvfft, axis=0
                    ), axes=0)
                )
            )
            return voltage, range_doppler, sample_freq, nfft, nvfft


    #initial data collection and axes:
    if plottype == 0:
        voltage, power, sample_freq, nfft = collect_data()
    elif plottype == 1:
        voltage, range_doppler, sample_freq, nfft, nvfft = collect_data()
    freq_axis = (sample_freq/2) * np.linspace(0, 1, nfft//2)
    range_axis = (3e8 * (ramp_time_ms/1000) * freq_axis) / \
        (2 * ((stop_f_GHz-start_f_GHz)*1e9))
    if plottype == 1:
        va = (3e8/((stop_f_GHz+start_f_GHz)*(1e9/2))) / (4*ramp_time_ms/1000)
        v_axis = np.linspace(-va, va, nvfft)

    #initial plot setup:
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes()
    if plottype == 0:
        ax.set_title("Range", weight='bold')
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Signal Power (dB)")
        rline,= ax.plot([], [], color='b')
        ax.set_xlim(range_axis[0]-1, range_axis[-1]+1)
        ax.set_ylim(-70, -10)
    elif plottype == 1:
        ax.set_title("Power Spectral Density Range-Doppler Plot")
        ax.set_xlabel("Doppler Velocity (m/s)")
        ax.set_ylabel("Range (m)")

        XX, YY = np.meshgrid(v_axis, range_axis)
        mesh = ax.pcolormesh(XX, YY, np.zeros_like(XX), vmin = -30, vmax = 10, cmap='terrain')
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Power Spectral Density (dB)")
    fig.show()

    exited_by_user = False
    def on_close(event=None):
        global exited_by_user
        global sweep_delay
        exited_by_user = True
        sweep_delay = 0.25 #need plot to close fast after x
        fig.canvas.stop_event_loop() #unpause if needed
        print("Closed by user. Exiting.")

    root = fig.canvas.manager.window
    root.protocol("WM_DELETE_WINDOW", on_close)

    #main loop
    n = 0
    condition = lambda n: (
        (True if (args.numsweeps is None) else (n < args.numsweeps)) and
        (not exited_by_user)
    )
    while condition(n):
        #recalculate for all subsequent:
        if n != 0:
            if plottype == 0:
                voltage, power, sample_freq, nfft = collect_data()
            elif plottype == 1:
                voltage, range_doppler, sample_freq, nfft, nvfft = collect_data()

        #execute plotting
        if plottype == 0:
            rline.set_data(range_axis, power)
        else:
            mesh.set_array(range_doppler.T)
        
        #update and save plot
        plt.pause(sweep_delay)
        if savefigs:
            fig.savefig(str(fig_path)+f"_{n}.png", bbox_inches='tight')

        #save data
        if savedata:
            output_data = {}
            #radar info
            output_data["sample_freq"] = sample_freq
            #scan params
            output_data["start_f_GHz"] = start_f_GHz
            output_data["stop_f_GHz"] = stop_f_GHz
            output_data["ramp_time_ms"] = ramp_time_ms
            #scan mode
            output_data["mode"] = mode_str_to_int[mode]
            #dsp window
            output_data["window"] = window_str_to_int[args.window]
            #unambiguous velocity (calculated from constants)
            if plottype == 1:
                output_data["va"] = va
            #dims and axes
            output_data["ramp_samples"] = voltage.shape[-1]
            if mode == "AUTOTRI":
                output_data["num_ramps"] = voltage.shape[0]
                output_data["total_samples"] = voltage.shape[-1]*voltage.shape[0]
            else:
                output_data["total_samples"] = voltage.shape[-1]
            output_data["time_axis"] = \
                (np.arange(output_data["total_samples"])*(1/sample_freq)).reshape(voltage.shape)
            output_data["nfft"] = nfft
            output_data["freq_axis"] = freq_axis
            output_data["range_axis"] = range_axis
            if plottype == 1:
                output_data["nvfft"] = nvfft
                output_data["v_axis"] = v_axis
            #data
            output_data["voltage"] = voltage
            if plottype == 0:
                output_data["power"] = power
            elif plottype == 1:
                output_data["range_doppler"] = range_doppler
            
            np.savez(str(data_path)+f"_{n}.npz", **output_data)

        n += 1

    if not exited_by_user:
        print("Finished plotting. Exiting.")
    plt.close(fig)
