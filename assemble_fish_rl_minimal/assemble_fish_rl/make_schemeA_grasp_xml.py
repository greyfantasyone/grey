from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


JOINT_DAMPING = {
    "Joint_1": "0.20",
    "Joint_2": "0.30",
    "Joint_3": "0.30",
    "Joint_4": "0.25",
}


def indent(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


TAIL_BODY_NAMES = {"Link_1", "Link_2", "Link_3", "Link_4"}


def remove_legacy_target(worldbody: ET.Element) -> None:
    to_remove = []
    for child in list(worldbody):
        if child.tag != "body":
            continue
        if child.attrib.get("name") == "base_link":
            continue
        has_freejoint = child.find("./freejoint") is not None
        geom = child.find("./geom")
        if has_freejoint and geom is not None and geom.attrib.get("type") == "cylinder":
            to_remove.append(child)
    for child in to_remove:
        worldbody.remove(child)


def add_task_objects(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("XML 中未找到 <worldbody>")

    remove_legacy_target(worldbody)

    base_body = worldbody.find("./body[@name='base_link']")
    if base_body is None:
        raise ValueError("XML 中未找到 base_link")

    if base_body.find("./site[@name='head_proxy']") is None:
        base_body.append(
            ET.Element(
                "site",
                {
                    "name": "head_proxy",
                    "pos": "0.10 0 0.0",
                    "size": "0.012",
                    "rgba": "0 0.6 1 1",
                },
            )
        )

    if worldbody.find("./body[@name='target_body']") is None:
        target_body = ET.Element(
            "body",
            {
                "name": "target_body",
                "pos": "-0.46 0.00 0.00",
            },
        )
        target_body.append(
            ET.Element(
                "geom",
                {
                    "name": "target_geom",
                    "type": "sphere",
                    "size": "0.025",
                    "rgba": "0.85 0.20 0.20 1",
                    "contype": "0",
                    "conaffinity": "0",
                },
            )
        )
        target_body.append(
            ET.Element(
                "site",
                {
                    "name": "target_site",
                    "pos": "0 0 0",
                    "size": "0.015",
                    "rgba": "1 0.1 0.1 1",
                },
            )
        )
        worldbody.append(target_body)

    if worldbody.find("./camera[@name='grasp_side']") is None:
        worldbody.append(
            ET.Element(
                "camera",
                {
                    "name": "grasp_side",
                    "pos": "0.55 -0.95 0.35",
                    "xyaxes": "1 0 0 0 0.35 0.94",
                },
            )
        )
    if worldbody.find("./camera[@name='grasp_top']") is None:
        worldbody.append(
            ET.Element(
                "camera",
                {
                    "name": "grasp_top",
                    "pos": "0.05 0.00 0.90",
                    "xyaxes": "1 0 0 0 1 0",
                },
            )
        )


ELLIPSOID_DEFAULTS = {
    "fitscale": "0.90",
    "fluidcoef": "0.50 0.15 1.20 0.10 0.10",
}


def add_tail_ellipsoid_fluid_geoms(root: ET.Element) -> None:
    for body in root.findall(".//body"):
        body_name = body.attrib.get("name", "")
        if body_name not in TAIL_BODY_NAMES:
            continue
        existing = any(geom.attrib.get("name", "").endswith("_fluid") for geom in body.findall("geom"))
        if existing:
            continue
        for geom in body.findall("geom"):
            if geom.attrib.get("type") == "mesh" and "mesh" in geom.attrib:
                mesh_name = geom.attrib["mesh"]
                fluid_geom = ET.Element(
                    "geom",
                    {
                        "name": f"{body_name}_fluid",
                        "type": "ellipsoid",
                        "mesh": mesh_name,
                        "fitscale": ELLIPSOID_DEFAULTS["fitscale"],
                        "fluidshape": "ellipsoid",
                        "fluidcoef": ELLIPSOID_DEFAULTS["fluidcoef"],
                        "rgba": "0 0 0 0",
                        "contype": "0",
                        "conaffinity": "0",
                        "group": "3",
                    },
                )
                body.append(fluid_geom)
                break


def patch_joint_damping(root: ET.Element) -> None:
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name")
        if name in JOINT_DAMPING and "damping" not in joint.attrib:
            joint.set("damping", JOINT_DAMPING[name])


def build_xml(src_xml: Path, dst_xml: Path, *, tail_ellipsoid_fluid: bool = False) -> None:
    tree = ET.parse(src_xml)
    root = tree.getroot()

    add_task_objects(root)
    patch_joint_damping(root)
    if tail_ellipsoid_fluid:
        add_tail_ellipsoid_fluid_geoms(root)

    indent(root)
    dst_xml.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从基础 XML 生成方案 A 的尾部抓取 XML")
    parser.add_argument("--src", type=Path, default=Path("phy3.02_with_ee.xml"))
    parser.add_argument("--dst", type=Path, default=Path("phy3.02_schemeA_grasp.xml"))
    parser.add_argument(
        "--tail-ellipsoid-fluid",
        action="store_true",
        help="给尾部 4 段额外加隐藏 ellipsoid fluid geoms，强化水动力耦合",
    )
    args = parser.parse_args()
    build_xml(args.src, args.dst, tail_ellipsoid_fluid=args.tail_ellipsoid_fluid)
    print(f"[OK] 写入: {args.dst}")
