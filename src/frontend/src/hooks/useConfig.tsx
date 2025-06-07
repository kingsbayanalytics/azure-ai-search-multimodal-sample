import { useEffect, useState } from "react";
import { listIndexes } from "../api/api";
import { OpenAIAPIMode } from "../api/models";

export interface SearchConfig {
    openai_api_mode: OpenAIAPIMode;
}

export default function useConfig() {
    const [config, setConfig] = useState<SearchConfig>({
        openai_api_mode: OpenAIAPIMode.ChatCompletions
        // use_streaming: true, // Removed, always true
        // use_prompt_flow: true // Removed, always true
    });

    const [indexes, setIndexes] = useState<string[]>([]);

    useEffect(() => {
        const fetchIndexes = async () => {
            const indexes = await listIndexes();
            setIndexes(indexes);
        };

        fetchIndexes();
    }, []);

    return { config, setConfig, indexes };
}
