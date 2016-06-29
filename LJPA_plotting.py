import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cst

from Calc import Calc

class Trends(Calc):
    """docstring for Trends"""
    def __init__(self, amp):
        Calc.__init__(self,amp)



    def resonance_plot(self, span=8e9, redline=False, xlim=None):
        """
        Return a figure with two plots showing the amplifier real impedance vs.
        frequency and imaginary impedance vs. frequency.

        Parameters
        ----------
        span : float, optional
            The frequency span over which the figure is plotted.
        redline : bool, optional
            If true shows lines at 50 ohms real impedance and 0 ohms imaginary
            impedance.
        xlim : list, optional
            two value list [xmin, xmax] to specify plot limits. If None, the
            function plots a span around the resonance frequency.
        """

        if not xlim:
            f0 = self.amp.resonance_frequency()
            fmin = (f0 - span/2)/1e9
            fmax = (f0 + span/2)/1e9
        else:
            fmin = xlim[0]
            fmax = xlim[1]
        f = np.linspace(fmin,fmax,1e3)
        real_impedance = np.real(self.amp.impedance(f*1e9))
        imag_impedance = np.imag(self.amp.impedance(f*1e9))

        ri_plot_max = max(real_impedance) + (max(real_impedance) - min(real_impedance))*0.1
        ri_plot_min = min(real_impedance) - (max(real_impedance) - min(real_impedance))*0.1

        fig = plt.subplots(figsize = (8,5))
        ax = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)

        if redline:
            ax.axhline(-50, color = 'red', linestyle = '--' )
            ax2.axhline(0, color = 'red', linestyle = '--' )

        ax.plot(f,real_impedance)
        ax2.plot(f,imag_impedance)

        ax.set_xlim([fmin,fmax])
        ax.set_ylim([ri_plot_min,ri_plot_max])
        ax2.set_xlim([fmin,fmax])

        ax2.set_xlabel('Frequency (GHz)', fontsize = 18)
        ax.set_ylabel('Resistance($\Omega$)', fontsize = 18)
        ax2.set_ylabel('Reactance($\Omega$)', fontsize = 18)

        ax.set_title(self.format_amp_inputs())

        return fig



    def phi_ac_gain_line_plot(self, xmin = 0.01, xmax = 0.5,
                                points = 1e2,
                                redline = False):
        """
        Return a figure with a plot showing the gain vs. phi_ac with
        all other parameters constant

        Parameters
        ----------
        xmin : float, optional
            minimum plotted phi_ac in Weber
        xmax : float, optional
            maximum plotted phi_ac in Weber
        point : float, optional
            number of plotted points
        redline : bool, optional
            If true shows line at maximum gain
        """

        backup_phi_ac = self.amp.phi_ac
        phi_ac = np.linspace(xmin, xmax, points)
        max_gain = []
        for value in phi_ac:
            self.amp.phi_ac = value
            max_gain.append(self.amp.find_max_gain())
        self.amp.phi_ac = backup_phi_ac

        max_phi_ac = phi_ac[max_gain.index(max(max_gain))]

        fig = plt.subplots(figsize = (8,5))
        ax = plt.subplot(1,1,1)

        if redline:
            ax.axvline(max_phi_ac,color = 'red', linestyle = '--')

        ax.plot(phi_ac,max_gain)

        ax.set_xlim([xmin, xmax])

        ax.set_xlabel('$\phi_{AC}$($\phi_0$)', fontsize = 18)
        ax.set_ylabel('Max Gain (dB)', fontsize = 18)

        ax.set_title(self.format_amp_inputs(block=['phi_s','theta_p', 'phi_ac']))

        return fig



    def phi_dc_f0_line_plot(self, xmin = -0.1, xmax = 0.4,
                                points = 1e2):
        """
        Return a figure with a plot showing the squid resonance frequency
        vs. phi_dc with all other parameters constant

        Parameters
        ----------
        xmin : float, optional
            minimum plotted phi_ac in Weber
        xmax : float, optional
            maximum plotted phi_ac in Weber
        points : float, optional
            number of plotted points
        """

        backup_phi_dc = self.amp.phi_dc
        phi_dc =  np.linspace(xmin, xmax, points)
        f_0 = []
        for value in phi_dc:
            self.amp.phi_dc = value
            f_0.append(self.amp.resonance_frequency()/1e9)
        self.amp.phi_dc = backup_phi_dc

        fig = plt.subplots(figsize = (8,5))
        ax = plt.subplot(1,1,1)

        ax.plot(phi_dc,f_0)

        ax.set_xlim([xmin,xmax])

        ax.set_xlabel('$\phi_{DC}$($\phi_0$)', fontsize = 18)
        ax.set_ylabel('$f_0$(GHz)', fontsize = 18)
        ax.set_title(self.format_amp_inputs(block=['phi_s', 'theta_p', 'phi_dc']))

        return fig



    def phi_dc_f0_gain_plot(self, xmin=-0.05, xmax = 0.3,
                            ymin=5, ymax=10,
                            pointsx=1e2, pointsy=1e2,
                            vmax=None):
        """
        Return a figure with a 2d temperature plot showing how the gain varies
        with frequency and phi_dc

        Parameters
        ----------
        xmin : float, optional
            minimum plotted phi_dc in Weber
        xmax : float, optional
            maximum plotted phi_dc in Weber
        pointsx : float, optional
            number of plotted phi_dc points
        pointsy : float, optional
            number of plotted frequency points
        vmax : float, optional
            maximum gain plotted. If None data values determine the range
        """

        backup_phi_dc = self.amp.phi_dc
        phi_dc = np.linspace(xmin,xmax,pointsx)
        f = np.linspace(ymin,ymax,pointsy)
        gain =[]
        for freq in f:
            for mag in phi_dc:
                self.amp.phi_dc = mag
                g = 10*np.log10(abs(self.amp.reflection(freq*1e9))**2.)
                gain.append(g)
        self.amp.phi_dc = backup_phi_dc

        gain = np.flipud(np.asarray(gain).reshape((pointsy,pointsx)))

        fig = plt.subplots(figsize = (8,5))
        ax = plt.subplot(1,1,1)

        aspect = (xmax - xmin)/(ymax - ymin)
        p = ax.imshow(gain, cmap=plt.get_cmap('plasma'), interpolation='none',
                        extent=[xmin,xmax,ymin,ymax], aspect=aspect, vmax=vmax)
        plt.colorbar(p).set_label(label='Gain (dB)',size=18)

        ax.set_xlabel('$\Phi_{DC} (\phi_0)$',fontsize=18)
        ax.set_ylabel('Signal Frequency (GHz)',fontsize=18)
        ax.set_title(self.format_amp_inputs(block=['phi_s', 'theta_p', 'phi_dc']))

        return fig



    def phi_dc_phi_ac_gain_plot(self, freq = 6., xmin=0.1, xmax = 0.4,
                            ymin=0.01, ymax=0.3,
                            pointsx=1e2, pointsy=1e2,
                            vmax=None):
        """
        Return a figure with a 2d temperature plot showing how the gain varies
        with phi_ac and phi_dc

        Parameters
        ----------
        freq : float, optional
            frequency of measurement in GHz
        xmin : float, optional
            minimum plotted phi_dc in Weber
        xmax : float, optional
            maximum plotted phi_dc in Weber
        pointsx : float, optional
            number of plotted phi_dc points
        pointsy : float, optional
            number of plotted phi_ac points
        vmax : float, optional
            maximum gain plotted. If None data values determine the range
        """

        backup_phi_dc = self.amp.phi_dc
        backup_phi_ac = self.amp.phi_ac
        phi_dc = np.linspace(xmin, xmax, pointsx)
        phi_ac = np.linspace(ymin, ymax, pointsy)
        gain =[]
        for amp_ac in phi_ac:
            for amp_dc in phi_dc:
                self.amp.phi_dc = amp_dc
                self.amp.phi_ac = amp_ac
                g = 10*np.log10(abs(self.amp.reflection(freq*1e9))**2.)
                gain.append(g)
        self.amp.phi_dc = backup_phi_dc
        self.amp.phi_ac = backup_phi_ac

        gain = np.flipud(np.asarray(gain).reshape((pointsy,pointsx)))

        fig = plt.subplots(figsize = (8,5))
        ax = plt.subplot(1,1,1)
        aspect = (xmax - xmin)/(ymax - ymin)
        p = ax.imshow(gain, cmap=plt.get_cmap('plasma'), interpolation='none',
                        extent=[xmin,xmax,ymin,ymax], aspect=aspect, vmax=vmax)
        plt.colorbar(p).set_label(label='Gain (dB)',size=18)

        plt.xlabel('$\Phi_{DC} (\phi_0)$',fontsize=18)
        plt.ylabel('$\Phi_{AC}(\phi_0)$',fontsize=18)
        plt.title(self.format_amp_inputs(block=['phi_s','theta_p', 'phi_ac',
                                                'phi_dc']) + ', f = {0} GHz'.format(freq))

        return fig



    def phi_dc_f0_phase_plot(self, xmin=-0.05, xmax = 0.3,
                            ymin=5, ymax=10,
                            pointsx=1e2, pointsy=1e2):
        """
        Return a figure with a 2d temperature plot showing how the gain varies
        with frequency and phi_dc

        Parameters
        ----------
        xmin : float, optional
            minimum plotted phi_dc in Weber
        xmax : float, optional
            maximum plotted phi_dc in Weber
        pointsx : float, optional
            number of plotted phi_dc points
        pointsy : float, optional
            number of plotted frequency points
        """

        backup_phi_dc = self.amp.phi_dc
        phi_dc = np.linspace(xmin,xmax,pointsx)
        f = np.linspace(ymin,ymax,pointsy)
        phase =[]
        for freq in f:
            for mag in phi_dc:
                self.amp.phi_dc = mag
                ph = np.angle(self.amp.reflection(freq*1e9))
                phase.append(ph)
        self.amp.phi_dc = backup_phi_dc

        phase = np.flipud(np.asarray(phase).reshape((pointsy,pointsx)))

        # with sns.color_palette("husl",314):
        fig = plt.subplots(figsize = (8,5))
        ax = plt.subplot(1,1,1)

        aspect = (xmax - xmin)/(ymax - ymin)

        p = ax.imshow(phase, interpolation='none', cmap='seismic',
                        extent=[xmin,xmax,ymin,ymax], aspect=aspect)
        plt.colorbar(p).set_label(label='Phase(rad)',size=18)

        ax.set_xlabel('$\Phi_{DC} (\phi_0)$',fontsize=18)
        ax.set_ylabel('Signal Frequency (GHz)',fontsize=18)
        ax.set_title(self.format_amp_inputs(block=['theta_p', 'phi_dc']))

        return fig



    def phi_dc_phi_ac_bandwidth_plot(self, freq = 6., xmin=0.1, xmax = 0.4,
                            ymin=0.01, ymax=0.3,
                            pointsx=1e2, pointsy=1e2,
                            vmax=None):
        """
        Return a figure with a 2d temperature plot showing how the gain varies
        with phi_ac and phi_dc

        Parameters
        ----------
        freq : float, optional
            frequency of measurement in GHz
        xmin : float, optional
            minimum plotted phi_dc in Weber
        xmax : float, optional
            maximum plotted phi_dc in Weber
        pointsx : float, optional
            number of plotted phi_dc points
        pointsy : float, optional
            number of plotted phi_ac points
        """

        backup_phi_dc = self.amp.phi_dc
        backup_phi_ac = self.amp.phi_ac
        phi_dc = np.linspace(xmin, xmax, pointsx)
        phi_ac = np.linspace(ymin, ymax, pointsy)
        bw =[]
        for amp_ac in phi_ac:
            for amp_dc in phi_dc:
                self.amp.phi_dc = amp_dc
                self.amp.phi_ac = amp_ac
                bandwidth = self.amp.find_reflection_fwhm()/1e6
                bw.append(bandwidth)
        self.amp.phi_dc = backup_phi_dc
        self.amp.phi_ac = backup_phi_ac

        bw = np.flipud(np.asarray(bw).reshape((pointsy,pointsx)))
        fig = plt.subplots(figsize = (8,5))
        ax = plt.subplot(1,1,1)
        aspect = (xmax - xmin)/(ymax - ymin)
        p = ax.imshow(bw, cmap=plt.get_cmap('plasma'), interpolation='none',
                        extent=[xmin,xmax,ymin,ymax], aspect=aspect, vmax=vmax)
        cs = plt.contour(np.flipud(bw), levels=[50], extent=[xmin,xmax,ymin,ymax],
                    colors='w',)
        # plt.clabel(cs,inline=1, fontsize=10, fmt='%.0f' )
        plt.colorbar(p).set_label(label='Bandwidth (MHz)',size=18)

        plt.xlabel('$\Phi_{DC} (\phi_0)$',fontsize=18)
        plt.ylabel('$\Phi_{AC}(\phi_0)$',fontsize=18)
        plt.title(self.format_amp_inputs(block=['phi_s','theta_p', 'phi_ac',
                                                'phi_dc']))

        return fig



    def phi_dc_phi_ac_gain_bandwidthcontours_plot(self, freq = 6., xmin=0.1, xmax = 0.4,
                            ymin=0.01, ymax=0.3,
                            pointsx=1e2, pointsy=1e2,
                            vmax=None):
        """
        Return a figure with a 2d temperature plot showing how the gain varies
        with phi_ac and phi_dc

        Parameters
        ----------
        freq : float, optional
            frequency of measurement in GHz
        xmin : float, optional
            minimum plotted phi_dc in Weber
        xmax : float, optional
            maximum plotted phi_dc in Weber
        pointsx : float, optional
            number of plotted phi_dc points
        pointsy : float, optional
            number of plotted phi_ac points
        vmax : float, optional
            maximum gain plotted. If None data values determine the range
        """

        backup_phi_dc = self.amp.phi_dc
        backup_phi_ac = self.amp.phi_ac
        phi_dc = np.linspace(xmin, xmax, pointsx)
        phi_ac = np.linspace(ymin, ymax, pointsy)
        gain =[]
        bw =[]
        for amp_ac in phi_ac:
            for amp_dc in phi_dc:
                self.amp.phi_dc = amp_dc
                self.amp.phi_ac = amp_ac
                g = 10*np.log10(abs(self.amp.reflection(freq*1e9))**2.)
                bandwidth = self.amp.find_reflection_fwhm()/1e6
                gain.append(g)
                bw.append(bandwidth)
        self.amp.phi_dc = backup_phi_dc
        self.amp.phi_ac = backup_phi_ac

        gain = np.flipud(np.asarray(gain).reshape((pointsy,pointsx)))
        bw = np.flipud(np.asarray(bw).reshape((pointsy,pointsx)))

        fig = plt.subplots(figsize = (8,5))
        ax = plt.subplot(1,1,1)
        aspect = (xmax - xmin)/(ymax - ymin)
        p = ax.imshow(gain, cmap=plt.get_cmap('plasma'), interpolation='none',
                        extent=[xmin,xmax,ymin,ymax], aspect=aspect, vmax=vmax)
        cs = plt.contour(np.flipud(bw), levels=[50], extent=[xmin,xmax,ymin,ymax],
                    colors='w',)
        plt.colorbar(p).set_label(label='Gain (dB)',size=18)

        plt.xlabel('$\Phi_{DC} (\phi_0)$',fontsize=18)
        plt.ylabel('$\Phi_{AC}(\phi_0)$',fontsize=18)
        plt.title(self.format_amp_inputs(block=['phi_s','theta_p', 'phi_ac',
                                                'phi_dc']) + ', f = {0} GHz'.format(freq))

        return fig



    def phi_s_josephson_inductance_line_plot(self, xmin = 0.01, xmax = 1e1,
                                pointsx = 1e2, freq=6e9):
        """
        Return a figure with a plot showing the gain vs. phi_ac with
        all other parameters constant

        Parameters
        ----------
        xmin : float, optional
            minimum plotted phi_ac in Weber
        xmax : float, optional
            maximum plotted phi_ac in Weber
        points : float, optional
            number of plotted points
        """

        backup_phi_s = self.amp.phi_s
        phi_s = np.linspace(xmin, xmax, pointsx)

        L_j = []
        power = []
        for value in phi_s:
            self.amp.phi_s = value
            L_j.append(self.amp.josephson_inductance())
            average_v_squared = 0.5*(cst.h/2./cst.e)**2*value**2*freq**2
            p = 10*np.log10(average_v_squared/abs(self.amp.squid_impedance(freq))/0.001)
            power.append(p)
        self.amp.phi_s = backup_phi_s



        fig = plt.subplots(figsize = (8,5))
        ax = plt.subplot(1,1,1)
        print 'max power =  ', max(power)
        ax.plot(power, L_j)

        ax.set_xlim([power[0], power[-1]])

        ax.set_xlabel('Signal Power (dBm)', fontsize = 18)
        ax.set_ylabel('$\mathrm{{L}}_\mathrm{{j}}$(H)', fontsize = 18)

        ax.set_title(self.format_amp_inputs(block=['theta_p','phi_s']) + ', f = {0} GHz'.format(freq/1e9), y=1.03)

        return fig



    def phi_s_pumpistor_inductance_line_plot(self, xmin = 0.01, xmax = 1e1,
                                pointsx = 1e2, freq=6e9):
        """

        Parameters
        ----------
        xmin : float, optional
            minimum plotted phi_ac in Weber
        xmax : float, optional
            maximum plotted phi_ac in Weber
        points : float, optional
            number of plotted points
        """

        backup_phi_s = self.amp.phi_s
        phi_s = np.linspace(xmin, xmax, pointsx)

        L_p_real = []
        L_p_imag = []
        power = []
        for value in phi_s:
            self.amp.phi_s = value
            L_p_real.append(np.real(self.amp.pumpistor_inductance()))
            L_p_imag.append(np.imag(self.amp.pumpistor_inductance()))
            average_v_squared = 0.5*(cst.h/2./cst.e)**2*value**2*freq**2
            p = 10*np.log10(average_v_squared/abs(self.amp.squid_impedance(freq))/0.001)
            power.append(p)
        self.amp.phi_s = backup_phi_s

        fig, (ax1, ax2) = plt.subplots(2, 1,figsize = (7*8./5.,7))
        ax1.plot(power, L_p_real)
        ax1.set_xlim([power[0], power[-1]])
        ax1.set_ylabel(r'Real $\mathrm{{L}}_\mathrm{{p}}$(H)', fontsize = 18)

        ax2.plot(power, L_p_imag)
        ax2.set_xlim([power[0], power[-1]])
        ax2.set_xlabel('Signal Power (dBm)', fontsize = 18)
        ax2.set_ylabel('Imaginary $\mathrm{{L}}_\mathrm{{p}}$(H)', fontsize = 18)

        ax1.set_title(self.format_amp_inputs(block=['theta_p','phi_s']) + ', f = {0} GHz'.format(freq/1e9), y=1.08)

        return fig



    def phi_dc_phi_ac_L_p_plot(self, freq = 6., xmin=0.01, xmax = 0.5,
                            ymin=0.01, ymax=0.5,
                            pointsx=1e2, pointsy=1e2,
                            vmax=None):
        """
        Return a figure with a 2d temperature plot showing how L_p varies
        with phi_ac and phi_dc

        Parameters
        ----------
        freq : float, optional
            frequency of measurement in GHz
        xmin : float, optional
            minimum plotted phi_dc in Weber
        xmax : float, optional
            maximum plotted phi_dc in Weber
        pointsx : float, optional
            number of plotted phi_dc points
        pointsy : float, optional
            number of plotted phi_ac points
        """

        backup_phi_dc = self.amp.phi_dc
        backup_phi_ac = self.amp.phi_ac
        phi_dc = np.linspace(xmin, xmax, pointsx)
        phi_ac = np.linspace(ymin, ymax, pointsy)
        lp =[]
        for amp_ac in phi_ac:
            for amp_dc in phi_dc:
                self.amp.phi_dc = amp_dc
                self.amp.phi_ac = amp_ac
                pumpistor_inductance = abs(self.amp.pumpistor_inductance())
                lp.append(pumpistor_inductance)
        self.amp.phi_dc = backup_phi_dc
        self.amp.phi_ac = backup_phi_ac

        lp = np.flipud(np.asarray(lp).reshape((pointsy,pointsx)))
        fig = plt.subplots(figsize = (8,5))
        ax = plt.subplot(1,1,1)
        aspect = (xmax - xmin)/(ymax - ymin)
        p = ax.imshow(lp, cmap=plt.get_cmap('plasma'), interpolation='none',
                        extent=[xmin,xmax,ymin,ymax], aspect=aspect, vmax=vmax)
        # cs = plt.contour(np.flipud(bw), levels=[50], extent=[xmin,xmax,ymin,ymax],
        #             colors='w',)
        # plt.clabel(cs,inline=1, fontsize=10, fmt='%.0f' )
        plt.colorbar(p).set_label(label='$|L_p|$(H)',size=18)

        plt.xlabel('$\Phi_{DC} (\phi_0)$',fontsize=18)
        plt.ylabel('$\Phi_{AC}(\phi_0)$',fontsize=18)
        plt.title(self.format_amp_inputs(block=['phi_s','theta_p', 'phi_ac',
                                                'phi_dc']))

        return fig



    def gain_vs_phi_s_freq_plot(self, xmin=0.01, xmax = 10.,
                            ymin=4., ymax=10.,
                            pointsx=1e2, pointsy=1e2,
                            vmax=None):
        """
        Return a figure with a 2d temperature plot showing how the gain varies
        with power and phi_dc

        Parameters
        ----------
        xmin : float, optional
            minimum plotted phi_s in radians
        xmax : float, optional
            maximum plotted phi_s in radians
        ymin : float, optional
            minimum plotted frequency in GHz
        ymax : float, optional
            maximum plotted frequency in GHz
        pointsx : float, optional
            number of plotted phi_dc points
        pointsy : float, optional
            number of plotted phi_ac points
        vmax : float, optional
            maximum gain plotted. If None data values determine the range
        """

        backup_phi_s = self.amp.phi_s
        phi_s = np.linspace(ymin, ymax, pointsy)
        freq = np.linspace(xmin, xmax, pointsx)
        gain =[]
        dev = []
        for value in phi_s:
            for f in freq:
                self.amp.phi_s = value
                gain.append(10*np.log10(abs(self.amp.reflection(float(f)*1e9))**2.))
        for f in freq:
            dev.append(self.amp.find_1db_deviation_power(float(f)*1e9, unit='rad'))
        self.amp.phi_s = backup_phi_s
        gain = np.flipud(np.asarray(gain).reshape((pointsy,pointsx)))
        fig = plt.subplots(figsize = (8,5))
        ax = plt.subplot(1,1,1)
        aspect = (xmax-xmin)/(ymax - ymin)
        p = ax.imshow(gain, cmap=plt.get_cmap('plasma'), interpolation='none',
                        extent=[xmin,xmax,ymin,ymax], aspect=aspect, vmax=vmax)
        ax.plot(freq, dev, color='w', lw=3)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        plt.colorbar(p).set_label(label='Gain (dB)',size=18)
        plt.xlabel('Signal Frequency (GHz)',fontsize=18)
        plt.ylabel('Signal Phase amplitude (rad)',fontsize=18)
        plt.title(self.format_amp_inputs(block=['phi_s','theta_p']))

        return fig



    def phi_dc_phi_ac_impedance_plot(self, freq = 6., xmin=0.01, xmax = 0.5,
                            ymin=0.01, ymax=0.5,
                            pointsx=1e2, pointsy=1e2,
                            vmax=None, real=True):
        """
        Return a figure with a 2d temperature plot showing how the impedance of
        the LJPA varies with phi_ac and phi_dc

        Parameters
        ----------
        freq : float, optional
            frequency of measurement in GHz
        xmin : float, optional
            minimum plotted phi_dc in Weber
        xmax : float, optional
            maximum plotted phi_dc in Weber
        pointsx : float, optional
            number of plotted phi_dc points
        pointsy : float, optional
            number of plotted phi_ac points
        """

        backup_phi_dc = self.amp.phi_dc
        backup_phi_ac = self.amp.phi_ac
        phi_dc = np.linspace(xmin, xmax, pointsx)
        phi_ac = np.linspace(ymin, ymax, pointsy)
        ri = []
        ii = []
        for amp_ac in phi_ac:
            for amp_dc in phi_dc:
                self.amp.phi_dc = amp_dc
                self.amp.phi_ac = amp_ac
                real_impedance = np.real(self.amp.impedance(freq*1e9))
                imag_impedance = np.imag(self.amp.impedance(freq*1e9))
                ri.append(real_impedance)
                ii.append(imag_impedance)
        self.amp.phi_dc = backup_phi_dc
        self.amp.phi_ac = backup_phi_ac

        ri = np.flipud(np.asarray(ri).reshape((pointsy,pointsx)))
        ii = np.flipud(np.asarray(ii).reshape((pointsy,pointsx)))

        fig = plt.subplots(figsize = (8,5))
        ax = plt.subplot(1,1,1)
        aspect = (xmax - xmin)/(ymax - ymin)
        if real:
            p = ax.imshow(ri, cmap=plt.get_cmap('plasma'), interpolation='none',
                            extent=[xmin,xmax,ymin,ymax], aspect=aspect, vmax=vmax)
            plt.colorbar(p).set_label(label='Real Impedance (Ohms)',size=18)
        else:
            p = ax.imshow(ii, cmap=plt.get_cmap('plasma'), interpolation='none',
                            extent=[xmin,xmax,ymin,ymax], aspect=aspect, vmax=vmax)
            plt.colorbar(p).set_label(label='Imaginary Impedance (Ohms)',size=18)

        plt.xlabel('$\Phi_{DC} (\phi_0)$',fontsize=18)
        plt.ylabel('$\Phi_{AC}(\phi_0)$',fontsize=18)
        plt.title(self.format_amp_inputs(block=['phi_s','theta_p', 'phi_ac',
                                                'phi_dc'])+ ', f = {0} GHz'.format(freq), y=1.06)

        return fig



    def f_s_LJPA_impedance_line_plot(self, xmin = 4, xmax = 8, ymin = None, ymax=None,
                                points = 1e2, real=True, redline=True, acmin=0.04, acmax=0.08,
                                acpoints=5):
        """
        Return a figure with a plot showing the impedance
        vs. signal frequency with all other parameters constant, including
        the pump frequency (non-degenerate operation)

        Parameters
        ----------
        xmin : float, optional
            minimum plotted f_s in GHz
        xmax : float, optional
            maximum plotted f_s in GHz
        points : float, optional
            number of plotted points
        """

        backup_phi_ac = self.amp.phi_ac
        f_s =  np.linspace(xmin, xmax, points)
        phi_ac = np.linspace(acmin, acmax, acpoints)
        ri = []
        ii = []
        for value in phi_ac:
            self.amp.phi_ac = value
            ri.append(np.real(self.amp.impedance(f_s*1e9)))
            ii.append(np.imag(self.amp.impedance(f_s*1e9)))
            print 'running phi_ac = ', value
        self.amp.phi_ac = backup_phi_ac

        fig, ax = plt.subplots(figsize = (8,5))

        if real:
            if redline and self.amp.f_p != None:
                # ax.axvline(self.amp.f_p/2e9, color = 'red', linestyle = '--' )
                ax.axhline(-50, color = 'red', linestyle = '--')
            for i in np.arange(len(phi_ac)):
                ax.plot(f_s,ri[i], label = '$\Phi_\mathrm{{AC}}$ = {:0.2f} $\phi_0$'.format(phi_ac[i]))
            ax.set_ylabel('Real Impedance (Ohms)', fontsize = 18)
            plt.legend(loc='upper left')

        else:
            if redline and self.amp.f_p != None:
                # ax.axvline(self.amp.f_p/2e9, color = 'red', linestyle = '--' )
                ax.axhline(0, color = 'red', linestyle = '--')
            for i in np.arange(len(phi_ac)):
                ax.plot(f_s,ii[i], label = '$\Phi_\mathrm{{AC}}$ = {:0.2f} $\phi_0$'.format(phi_ac[i]))
            ax.set_ylabel('Imaginary Impedance (Ohms)', fontsize = 18)
            plt.legend()


        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        ax.set_xlabel('Signal Frequency (GHz)', fontsize = 18)
        ax.set_title(self.format_amp_inputs(block=['phi_s', 'theta_p', 'phi_ac']))

        return fig



    def f_s_LJPA_gain_line_plot(self, xmin = 4, xmax = 8, ymin = None, ymax=None,
                                points = 1e2, redline=True, acmin=0.04, acmax=0.08,
                                acpoints=5):
        """
        Return a figure with a plot showing the impedance
        vs. signal frequency with all other parameters constant, including
        the pump frequency (non-degenerate operation)

        Parameters
        ----------
        xmin : float, optional
            minimum plotted f_s in GHz
        xmax : float, optional
            maximum plotted f_s in GHz
        points : float, optional
            number of plotted points
        """

        backup_phi_ac = self.amp.phi_ac
        f_s =  np.linspace(xmin, xmax, points)
        phi_ac = np.linspace(acmin, acmax, acpoints)
        gain = []
        for value in phi_ac:
            self.amp.phi_ac = value
            gain.append(10*np.log10(abs(self.amp.reflection(f_s*1e9))**2.))
            print 'running phi_ac = ', value
        self.amp.phi_ac = backup_phi_ac

        fig, ax = plt.subplots(figsize = (8,5))

        for i in np.arange(len(phi_ac)):
            ax.plot(f_s,gain[i], label = '$\Phi_\mathrm{{AC}}$ = {:0.3f} $\phi_0$'.format(phi_ac[i]))
        ax.set_ylabel('Gain (dB)', fontsize = 18)
        plt.legend(loc='upper right')


        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        ax.set_xlabel('Signal Frequency (GHz)', fontsize = 18)
        ax.set_title(self.format_amp_inputs(block=['phi_s', 'theta_p', 'phi_ac']))

        return fig



    def gain_impedance(self, fmin=5, fmax=7, fpoints=1e2, gmin=None, gmax=None, rmin=None,
                        rmax=None, imin=None, imax=None, redline=True):
        f = np.linspace(fmin,fmax,fpoints)
        power = 10*np.log10(abs(self.amp.reflection(f*1e9))**2)
        z = self.amp.impedance(f*1e9)
        ri = np.real(z)
        ii = np.imag(z)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8,5))
        if redline:
            ax2.plot([f[0], f[-1]], [-50., -50.], color='r', linestyle='--')
            ax3.plot([f[0], f[-1]], [0., 0.], color='r', linestyle='--')
        ax1.plot(f, power)
        ax1.set_ylabel('Gain (dB)')
        ax1.set_ylim([gmin, gmax])
        ax2.plot(f, ri)

        ax2.set_ylabel('Real Imp. (Ohms)')
        ax2.set_ylim([rmin, rmax])
        ax3.plot(f,ii)

        ax3.set_ylim([imin, imax])
        ax3.set_ylabel('Imag. Imp. (Ohms)')
        ax3.set_xlabel('Signal Frequency (GHz)')
        ax1.set_title(self.format_amp_inputs())

        return fig
