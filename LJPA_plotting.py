import matplotlib.pyplot as plt
import numpy as np

class Trends(object):
    """docstring for Trends"""
    def __init__(self, amp):
        super(Trends, self).__init__()
        self.amp = amp
    def format_amp_inputs(self, block=['phi_s', 'theta_p']):
        """
        Return a string with the amplifier inputs formatted in readable units.

        Parameters
        ----------
        block : list, optional
            The amplifier values that should not be displayed. May include
                'C'
                'L_s'
                'I_c'
                'phi_s'
                'phi_dc'
                'phi_dc'
                'phi_ac'
                'theta_p'
        """
        # Get a list of variables parameters name and value
        params_name  = ['C','L_s','I_c','phi_s', 'phi_dc', 'phi_ac', 'theta_p']
        params_value = [self.amp.C, self.amp.L_s, self.amp.I_c, self.amp.phi_s,
                        self.amp.phi_dc, self.amp.phi_ac, self.amp.theta_p]

        values = []
        names  = []
        for param_name, param_value in zip(params_name, params_value):
            if param_name not in block:
                names.append(param_name)
                values.append(param_value)

        inputs = ''
        for name, value in zip(names, values):
            if inputs != '':
                inputs += ','
            if name == 'C':
                C = ' C = {:.1f} pF '.format(value/1e-12)
                inputs += C
            elif name == 'L_s':
                L_s = ' $L_s$ = {:.1f} pH '.format(value/1e-12)
                inputs += L_s
            elif name == 'I_c':
                I_c = ' $I_c$ = {:0.2f} $\mu$A '.format(value/1e-6)
                inputs += I_c
            elif name == 'phi_s':
                phi_s = ' $\Phi_s$ = {:0.2f} rad '.format(value)
                inputs += phi_s
            elif name == 'phi_dc':
                phi_dc = ' $\Phi_{{dc}}$ = {:0.2f} $\phi_0$ '.format(value)
                inputs += phi_dc
            elif name == 'phi_ac':
                phi_ac = ' $\Phi_{{ac}}$ = {:0.2f} $\phi_0$ '.format(value)
                inputs += phi_ac
            elif name == 'theta_p':
                theta_p = ' $ \theta_p$ = {:0.2f} rad '.format(value)
                inputs += theta_p
        return inputs

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
                            maxgain=None):
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
        maxgain : float, optional
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
                        extent=[xmin,xmax,ymin,ymax], aspect=aspect, vmax=maxgain)
        plt.colorbar(p).set_label(label='Gain (dB)',size=18)

        ax.set_xlabel('$\Phi_{DC} (\phi_0)$',fontsize=18)
        ax.set_ylabel('Frequency (GHz)',fontsize=18)
        ax.set_title(self.format_amp_inputs(block=['phi_s', 'theta_p', 'phi_dc']))
        return fig

    def phi_dc_phi_ac_gain_plot(self, freq = 6., xmin=0.1, xmax = 0.4,
                            ymin=0.01, ymax=0.3,
                            pointsx=1e2, pointsy=1e2):
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
                        extent=[xmin,xmax,ymin,ymax], aspect=aspect)
        plt.colorbar(p).set_label(label='Gain',size=18)

        plt.xlabel('$\Phi_{DC} (\phi_0)$',fontsize=18)
        plt.ylabel('$\Phi_{AC}(\phi_0)$',fontsize=18)
        plt.title(self.format_amp_inputs(block=['phi_s','theta_p', 'phi_ac',
                                                'phi_dc']) + ', f = {0} GHz'.format(freq))
        return fig
