import time

import pyvisa

import numpy as np
from math import floor, ceil

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
        "If omitted, figures will not be saved. "
        "Prefixes beginning with / or containing ':' "
        "as the second character (Windows) will be evaluated from root. "
        "Other prefixes will be evaluated from the current working directory.",
    )

    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    plt.ioff()

    args = parser.parse_args()

    start_f_GHz = args.start
    stop_f_GHz = args.stop
    mode = args.mode
    ramp_time_ms = args.ramptime

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
    
    if args.figprefix is None:
        savefigs = False
    else:
        savefigs = True
    if savefigs:
        file_path = Path(args.figprefix)
        if args.figprefix.endswith(('/', '\\')) or file_path.is_dir():
            raise ValueError(
                f"--figprefix '{args.figprefix}' appears to be a directory. "
                "Please provide a file prefix (e.g., 'folder/myplot')."
            )
        full_path = file_path.resolve() 
        figdir = full_path.parent

        # 5. Validate the directory
        if not figdir.exists():
            raise FileNotFoundError(f"The directory '{figdir}' does not exist.")
        if not figdir.is_dir():
            raise NotADirectoryError(f"The path '{figdir}' exists but is not a directory.")
    
    if not (args.numsweeps is None):
        if args.numsweeps <= 1:
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
        beats = (2*np.fft.fft(voltage, nfft, axis=-1)/ramp_samples)[...,0:nfft//2]

        if plottype == 0:
            power = 20*np.log10(np.abs(beats))
            if mode == 'AUTOTRI':
                power = np.mean(power, axis=0)
            return voltage, power, sample_freq, nfft
        elif plottype == 1:
            num_ramps = voltage.shape[0]
            nvfft = num_ramps
            range_doppler = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(beats, nvfft, axis=0), axes=0)))
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
    fig.show()
    ax = plt.axes()
    if plottype == 0:
        ax.set_title("Range", weight='bold')
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Signal Power (dB)")
        rline,= ax.plot([], [], color='b')
    elif plottype == 1:
        ax.set_title("Power Spectral Density Range-Doppler Plot")
        ax.set_xlabel("Doppler Velocity (m/s)")
        ax.set_ylabel("Range (m)")

        XX, YY = np.meshgrid(v_axis, range_axis)
        mesh = ax.pcolormesh(XX, YY, np.zeros_like(XX), vmin = -30, vmax = 10, cmap='terrain')
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Power Spectral Density (dB)")

    exited_by_user = False
    def on_close(event=None):
        global exited_by_user
        exited_by_user = True
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
        ax.relim()
        fig.canvas.draw()
        fig.canvas.flush_events()
        if savefigs:
            fig.savefig(str(full_path)+f"_{n}.png", bbox_inches='tight')
        plt.pause(sweep_delay)

        n += 1

    print("Finished plotting. Exiting.")
    plt.close(fig)