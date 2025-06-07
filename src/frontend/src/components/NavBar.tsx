import { useState } from "react";

import { Button, Tooltip } from "@fluentui/react-components";
import { ChatAddRegular } from "@fluentui/react-icons";
import { Hamburger, NavDrawer, NavDrawerHeader } from "@fluentui/react-nav-preview";

import { Chat } from "../api/models";
import "./NavBar.css";

interface Props {
    indexes: string[];
    chats: Chat[];
    onNewChat: () => void;
}

export const NavBar = ({ onNewChat }: Props) => {
    const [isOpen, setIsOpen] = useState(false);

    const getToolTipContent = () => {
        return isOpen ? "Close Settings" : "Open Settings";
    };

    return (
        <>
            <NavDrawer open={isOpen} type={"inline"} className="menu">
                <div className="menu-items">
                    <Button appearance="secondary" icon={<ChatAddRegular />} className="custom-menu-item new-chat" onClick={onNewChat}>
                        New Chat
                    </Button>
                </div>
            </NavDrawer>
            <NavDrawerHeader style={{ width: "25px" }}>
                <Tooltip content={getToolTipContent()} relationship="label">
                    <Hamburger onClick={() => setIsOpen(!isOpen)} />
                </Tooltip>
            </NavDrawerHeader>
        </>
    );
};
