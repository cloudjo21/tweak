from ner.evaluate.muc_evaluator import Entity

def filter_entities_by_pos(entity_tuples, eval_excluded_pos, pos_flatten, labels_flatten):
    
    remove_conds = []
    
    for ent in entity_tuples:
        ent_pos = pos_flatten[ent.start_offset:ent.end_offset+1]
        head_ent_pos = list(map(lambda x: x[0], ent_pos))
        remove_conds.append(True if set(head_ent_pos).issubset(set(eval_excluded_pos)) else False)
    
    for is_remove, ent in zip(remove_conds, entity_tuples):
        if is_remove:
            labels_flatten[ent.start_offset:ent.end_offset+1] = list(map(
                lambda x: 'O', labels_flatten[ent.start_offset:ent.end_offset+1]
            ))
            
    return labels_flatten
    