import cv2

import networkx as nx
import matplotlib.image as mpimg


from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator
from PIL import Image
from io import BytesIO
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def make_mol_png_file(mol,filename,legend='',border_values=[0,0,0]):
    d2d = Draw.MolDraw2DCairo(350,350)
    dopts = d2d.drawOptions()
    dopts.setBackgroundColour((1,1,1,1))
    dopts.legendFraction = 0.3
    dopts.legendFontSize = 45
    d2d.DrawMolecule(mol,legend=legend)
    d2d.FinishDrawing()
    bio = BytesIO(d2d.GetDrawingText())
    img = Image.open(bio)
    img.save(filename,"PNG")
    virat_img = cv2.imread(filename)
    borderoutput = cv2.copyMakeBorder(
        virat_img, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=border_values)
    cv2.imwrite(filename, borderoutput)

def get_bit_info(mol, nBits=2048, radius=1):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=nBits)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    fp = mfpgen.GetCountFingerprint(mol,additionalOutput=ao)
    bi = ao.GetBitInfoMap()
    return bi

def make_bit_png(mol,bit,bit_info,filename,legend='',border_values=[0,0,0]):
    d2d = Draw.MolDraw2DCairo(350,350)
    dopts=d2d.drawOptions()
    dopts.setBackgroundColour((1,1,1,1))
    dopts.legendFraction = 0.3
    dopts.legendFontSize = 20
    img = Draw.DrawMorganBit(mol, bit, bit_info, legend=legend, useSVG=False, drawOptions=dopts)
    img.save(filename, size=(350,350))
    virat_img = cv2.imread(filename)
    borderoutput = cv2.copyMakeBorder(
        virat_img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=border_values)
    cv2.imwrite(filename, borderoutput)

def make_bit_png_file(train_mols,bit,filename,legend="",border_values=[0,0,0]):
    for mol in train_mols:
        bit_info = get_bit_info(mol, nBits=2048, radius=1)
        if bit in bit_info.keys():
            break
    make_bit_png(mol, bit, bit_info,filename,legend=legend,border_values=border_values)

def extract_value(value, regression_case):
    """Extract value from tree node"""
    if regression_case:
        # for regressor, value is in a nested list
        return value[0][0]
    else:
        # for classifier, value is in a list
        return value[0]

def get_node_label(nxtree, i_node, regression_case):
    """Get a label for a tree node

    This is a general label for use in a legend.
    Several nodes can share the same label"""
    node_attr_dict = nxtree.nodes[i_node]
    feature = node_attr_dict["feature"]
    node_type = node_attr_dict["node_type"]

    if node_type == "split":
        label = f"split on {feature}"
    elif node_type == "leaf":
        if regression_case:
            label = "leaf"
        else:
            label = f"leaf {node_attr_dict['class_name']}"
    return label

def get_node_description(nxtree, i_node, regression_case):
    """Get a description for a tree node

    This should be unique for each node"""

    node_attr_dict = nxtree.nodes[i_node]
    label = node_attr_dict["label"]
    node_type = node_attr_dict["node_type"]
    description = f"{i_node}\n{label}"

    if node_type == "split":
        thr = node_attr_dict["threshold"]
        description = description + f"\nthreshold: {thr:.3g}"
    elif node_type == "leaf":
        if regression_case:
            value = node_attr_dict["value"]
            description = description + f"\nvalue: {value:.3g}"
    return description

def get_networkx_tree_from_sklearn(
    skl_decision_tree, train_mols, train_targets, feature_names=None, class_names=None,path=None
) -> nx.DiGraph:
    """Get Networkx tree from sklearn decision tree

    Args:
        skl_decision_tree: scikit-learn decision tree:
            DecisionTreeClassifier or DecisionTreeRegressor
        train_mols: list of RDKit mol objects for the train molecules
        train_targets: list of target values for the train molecules
        feature_names: optional list of feature names
        class_names: optional list of class names (for the classification case)

    Returns:
        a NetworkX digraph representing the decision tree
            with node attributes "node_type", "value", "label" and "description"
            and edge attribute "label"
    """
    assert isinstance(
        skl_decision_tree, (DecisionTreeRegressor, DecisionTreeClassifier)
    )
    tree = skl_decision_tree.tree_
    regression_case = isinstance(skl_decision_tree, DecisionTreeRegressor)

    nxtree = nx.DiGraph(regression_case=regression_case)
    for i_node in range(tree.node_count):
        feature_id = tree.feature[i_node]
        if feature_id >= 0:
            if feature_names is not None:
                feature = feature_names[feature_id]
            else:
                feature = str(feature_id)
            node_type = "split"
        else:
            feature = None
            node_type = "leaf"

        # copy node attributes
        node_attr_dict = {
            attr_name: getattr(tree, attr_name)[i_node]
            for attr_name in ["n_node_samples", "impurity", "threshold", "value"]
        }
        node_attr_dict["value"] = extract_value(
            node_attr_dict["value"], regression_case
        )
        if not regression_case:
            node_attr_dict["class_name"] = get_node_class_name(
                node_attr_dict["value"], class_names
            )
        filename=f'DT_tree/{i_node}.png'
        border_values=[0,0,0]
        if path is not None:
            if i_node in path:
                border_values=[0,255,0]

        if node_type == "leaf":
            leaf_value = node_attr_dict["value"]
            leaf_mol = train_mols[train_targets.index(leaf_value)]
            legend = f'Forudsigelse baseret paa\n dette molekyle: y={leaf_value:.2f}'
            make_mol_png_file(leaf_mol, filename, legend=legend, border_values=border_values)
        else:
            bit = int(feature)
            threshold = node_attr_dict["threshold"]
            if threshold.is_integer():
                threshold += 0.5
            legend = f"flere eller faerre end\n {threshold} af det her fragment?"
            make_bit_png_file(train_mols,bit,filename,legend=legend, border_values=border_values)
        img=mpimg.imread(filename)
        nxtree.add_node(i_node,  image=img, feature=feature, **node_attr_dict, node_type=node_type)

    # add edges
    for i_node in range(tree.node_count):
        node_children = (tree.children_left[i_node], tree.children_right[i_node])
        for i_child, edge_label in zip(node_children, ("<=", ">")):
            if i_child >= 0:
                if edge_label=="<=":
                    edge_label="færre"
                if edge_label==">":
                    edge_label="flere"
                nxtree.add_edge(i_node, i_child, label=edge_label)

    # add labels
    for i_node in nxtree.nodes:
        nxtree.nodes[i_node]["label"] = get_node_label(nxtree, i_node, regression_case)
        nxtree.nodes[i_node]["description"] = get_node_description(
            nxtree, i_node, regression_case
        )
    return nxtree
