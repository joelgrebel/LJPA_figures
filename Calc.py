
class Calc(object):

    def __init__(self, amp):
        """helper functions for LJPA plotting"""
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
        params_name  = ['C','L_s','I_c','phi_s', 'phi_dc', 'phi_ac', 'theta_p','f_p']
        params_value = [self.amp.C, self.amp.L_s, self.amp.I_c, self.amp.phi_s,
                        self.amp.phi_dc, self.amp.phi_ac, self.amp.theta_p, self.amp.f_p]

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
                L_s = ' $\mathrm{{L}}_\mathrm{{s}}$ = {:.1f} pH '.format(value/1e-12)
                inputs += L_s
            elif name == 'I_c':
                I_c = ' $\mathrm{{I}}_\mathrm{{c}}$ = {:0.2f} $\mu$A '.format(value/1e-6)
                inputs += I_c
            elif name == 'phi_s':
                phi_s = ' $\Phi_\mathrm{{s}}$ = {:0.2f} rad '.format(value)
                inputs += phi_s
            elif name == 'phi_dc':
                phi_dc = ' $\Phi_\mathrm{{dc}}$ = {:0.2f} $\phi_0$ '.format(value)
                inputs += phi_dc
            elif name == 'phi_ac':
                phi_ac = ' $\Phi_\mathrm{{ac}}$ = {:0.3f} $\phi_0$ '.format(value)
                inputs += phi_ac
            elif name == 'theta_p':
                theta_p = r' $ \theta_\mathrm{{p}}$ = {:0.2f} rad '.format(value)
                inputs += theta_p
            elif name == 'f_p':
                if value==None:
                    f_p = ' $ \mathrm{{f}}_\mathrm{{p}} = 2\mathrm{{f}}_\mathrm{{s}} $'
                else:
                    f_p = ' $ \mathrm{{f}}_\mathrm{{p}} = {:0.2f} $ GHz'.format(value/1e9)
                inputs += f_p
        return inputs
