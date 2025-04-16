import { useAtom } from 'jotai';
import { searchPanelOpenAtom } from '@/store/searchPanel';
import appConfig from '@/config.json';
import { DocSearchModal, useDocSearchKeyboardEvents } from '@docsearch/react';
import '@docsearch/css';
import { RootPortal } from './RootPortal';
// 调整后的导入方式
import { useRef } from'react';
type RefObject = React.RefObject;

export function SearchPanel() {
    const [isOpen, setIsOpen] = useAtom(searchPanelOpenAtom);
    // 创建 searchButtonRef
    const searchButtonRef: RefObject<HTMLButtonElement> = useRef(null);

    const onOpen = () => {
        setIsOpen(true);
    };
    const onClose = () => {
        setIsOpen(false);
    };

    useDocSearchKeyboardEvents({
        isOpen,
        onOpen,
        onClose,
        searchButtonRef
    });

    return (
        isOpen && (
            <RootPortal>
                <DocSearchModal
                    appId={appConfig.docSearch.appId}
                    apiKey={appConfig.docSearch.apiKey}
                    indexName={appConfig.docSearch.indexName}
                    initialScrollY={window.scrollY}
                    onClose={onClose}
                />
            </RootPortal>
        )
    );
}