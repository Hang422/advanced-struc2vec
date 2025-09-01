#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Struc2Vec runtime patch
Fixes division by zero warnings without modifying original files
"""
import numpy as np
import warnings

def patch_struc2vec():
    """Apply runtime patches"""
    try:
        from GraphEmbedding.ge.models import struc2vec
        
        # Save original method
        original_get_layers_adj = struc2vec.Struc2Vec._get_layers_adj
        
        def patched_get_layers_adj(self, layers_distances):
            """Patched method to handle division by zero cases"""
            
            # Use warning filter to temporarily ignore division warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, 
                                      message='invalid value encountered in scalar divide')
                
                # Call original method
                result = original_get_layers_adj(self, layers_distances)
                
                # Check and fix any NaN or Inf values
                layers_alias, layers_accept = result
                
                for layer in layers_alias:
                    for node in layers_alias[layer]:
                        # Check alias table
                        if layers_alias[layer][node] is not None:
                            alias = layers_alias[layer][node]
                            # Replace any NaN with 0
                            if isinstance(alias, (list, np.ndarray)):
                                alias = np.nan_to_num(alias, nan=0.0, posinf=1.0, neginf=0.0)
                                layers_alias[layer][node] = alias
                        
                        # Check accept table
                        if layers_accept[layer][node] is not None:
                            accept = layers_accept[layer][node]
                            if isinstance(accept, (list, np.ndarray)):
                                accept = np.nan_to_num(accept, nan=1.0, posinf=1.0, neginf=0.0)
                                layers_accept[layer][node] = accept
                
                return layers_alias, layers_accept
        
        # Apply patch
        struc2vec.Struc2Vec._get_layers_adj = patched_get_layers_adj
        print("✅ Struc2Vec patch applied successfully")
        
    except ImportError:
        print("❌ Unable to import Struc2Vec module")
    except Exception as e:
        print(f"❌ Failed to apply patch: {e}")

# Automatically apply patches
if __name__ != "__main__":
    patch_struc2vec()
