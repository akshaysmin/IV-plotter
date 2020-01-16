# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19, 2018

@author: Timofey Golubev

This contains everything used to read simulation parameters from file and defines a Params class,
an instance of which can be used to store the parameters.
"""

import math, constants as const, numpy as np

def is_positive(value, comment):   
    '''
    Checks if an input value is positive.
    Inputs:
        value:   the input value
        comment: this is used to be able to output an informative error message, 
                 if the input value is invalid
    '''
    
    if value <= 0:
        print(f"Non-positive input for {comment}\n Input was read as {value}.")
        raise ValueError("This input must be positive")
                    
def is_negative(value, comment):
    '''
    Checks if an input value is positive.
    Inputs:
        value:   the input value
        comment: this is used to be able to output an informative error message, 
                 if the input value is invalid
    '''
    
    if value >= 0:
        print(f"Non-positive input for {comment}\n Input was read as {value}.")
        raise ValueError("This input must be negative")

class Params():
    
    '''
    The Params class groups all of the simulation parameters parameters into a parameters object.
    Initialization of a Params instance, reads in the parameters from "parameters.inp" input file.  
    '''
    
    def __init__(self):
        
        try:
            parameters = open("parameters.inp", "r")
        except:
            print(f"Unable to open file parameters.inp")
            
        try:
            comment = parameters.readline()
            tmp = parameters.readline().split()
            self.L = float(tmp[0])  #note: floats in python are double precision
            comment = tmp[1]
            is_positive(self.L, comment)
                
            tmp = parameters.readline().split()
            self.N_LUMO = float(tmp[0])  
            comment = tmp[1]
            is_positive(self.N_LUMO, comment)
            
            tmp = parameters.readline().split()
            self.N_HOMO = float(tmp[0])  
            comment = tmp[1]
            is_positive(self.N_HOMO, comment)
            
            tmp = parameters.readline().split()
            self.Photogen_scaling = float(tmp[0])  
            comment = tmp[1]
            is_positive(self.Photogen_scaling, comment)
            
            tmp = parameters.readline().split()
            self.phi_a  = float(tmp[0]) 
            comment = tmp[1]
            is_positive(self.phi_a , comment)
            
            tmp = parameters.readline().split()
            self.phi_c = float(tmp[0])  
            comment = tmp[1]
            is_positive(self.phi_c, comment)
            
            tmp = parameters.readline().split()
            self.eps_active = float(tmp[0])  
            comment = tmp[1]
            is_positive(self.eps_active, comment)
            
            tmp = parameters.readline().split()
            self.p_mob_active = float(tmp[0])  
            comment = tmp[1]
            is_positive(self.p_mob_active, comment)
            
            tmp = parameters.readline().split()
            self.n_mob_active = float(tmp[0])  
            comment = tmp[1]
            is_positive(self.n_mob_active, comment)
            
            tmp = parameters.readline().split()
            self.mobil = float(tmp[0]) 
            comment = tmp[1]
            is_positive(self.mobil, comment)
            
            tmp = parameters.readline().split()
            self.E_gap = float(tmp[0]) 
            comment = tmp[1]
            is_positive(self.E_gap, comment)
            
            tmp = parameters.readline().split()
            self.active_CB = float(tmp[0]) 
            comment = tmp[1]
            is_negative(self.active_CB, comment)
            
            tmp = parameters.readline().split()
            self.active_VB = float(tmp[0]) 
            comment = tmp[1]
            is_negative(self.active_VB, comment)
            
            tmp = parameters.readline().split()
            self.WF_anode = float(tmp[0]) 
            comment = tmp[1]
            is_positive(self.WF_anode, comment)
            
            tmp = parameters.readline().split()
            self.WF_cathode = float(tmp[0]) 
            comment = tmp[1]
            is_positive(self.WF_cathode, comment)
            
            tmp = parameters.readline().split()
            self.k_rec = float(tmp[0]) 
            comment = tmp[1]
            is_positive(self.k_rec, comment)
            
            tmp = parameters.readline().split()
            self.dx = float(tmp[0]) 
            comment = tmp[1]
            is_positive(self.dx, comment)
            
            tmp = parameters.readline().split()
            self.Va_min= float(tmp[0]) 
            
            tmp = parameters.readline().split()
            self.Va_max = float(tmp[0]) 
            
            tmp = parameters.readline().split()
            self.increment = float(tmp[0]) 
            comment = tmp[1]
            is_positive(self.increment, comment)
            
            tmp = parameters.readline().split()
            self.w_eq = float(tmp[0]) 
            comment = tmp[1]
            is_positive(self.w_eq, comment)
            
            tmp = parameters.readline().split()
            self.w_i = float(tmp[0]) 
            comment = tmp[1]
            is_positive(self.w_i, comment)
            
            tmp = parameters.readline().split()
            self.tolerance_i  = float(tmp[0]) 
            comment = tmp[1]
            is_positive(self.tolerance_i , comment)
            
            tmp = parameters.readline().split()
            self.w_reduce_factor = float(tmp[0]) 
            comment = tmp[1]
            is_positive(self.w_reduce_factor, comment)
            
            tmp = parameters.readline().split()
            self.tol_relax_factor = float(tmp[0]) 
            comment = tmp[1]
            is_positive(self.tol_relax_factor, comment)
            
            tmp = parameters.readline().split()
            self.gen_rate_file_name = tmp[0] 
            
            # calculated parameters
            self.N = self.N_HOMO    
            self.num_cell = math.ceil(self.L/self.dx)  
            self.E_trap = self.active_VB + self.E_gap/2.0 # traps are assumed to be at 1/2 of the bandgap
            self.n1 = self.N_LUMO*np.exp(-(self.active_CB - self.E_trap)/const.Vt)
            self.p1 = self.N_HOMO*np.exp(-(self.E_trap - self.active_VB)/const.Vt)
            
        except:
            print(tmp)
            print("Invalid Input. Fix it and rerun")
            
    
    # The following functions are mostly to make the main code a bit more readable and obvious what
    # is being done.
        
    def reduce_w(self):
        '''
        Reduces the weighting factor (w) (used for linear mixing of old and new solutions) by w_reduce_factor
        which is defined in the input parameters
        '''
        self.w = self.w/self.w_reduce_factor
        
    def relax_tolerance(self):
        '''
        Relax the criterea for determining convergence of a solution by the tol_relax_factor. 
        This is sometimes necessary for hard to converge situations. 
        The relaxing of tolerance is done automatically when convergence issues are detected.
        '''
        self.tolerance = self.tolerance*self.tol_relax_factor
        
    def use_tolerance_eq(self):
        '''
        Use the convergence tolerance meant for equilibrium condition run. This tolerance is usually
        higher than the regular tolerance due the problem is more difficult to converge when simulating
        at 0 applied voltage.
        '''
        self.tolerance = self.tolerance_eq
        
    def use_tolerance_i(self):
        '''
        Use the initial convergence tolerance specified (before any relaxing of the tolerance).
        '''
        self.tolerance = self.tolerance_i
        
    def use_w_i(self):
        '''
        Use the initially specified weighting factor (w) (used for linear mixing of old and new solutions).
        '''
        self.w = self.w_i
        
    def use_w_eq(self):
        '''
        Use the weighting factor (w) (used for linear mixing of old and new solutions) for the equilibrium 
        condition run. This is usually lower than the regular w due the problem is more difficult to 
        converge when simulating at 0 applied voltage.
        '''
        self.w = self.w_eq
        
